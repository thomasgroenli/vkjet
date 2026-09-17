# vkjet

Fit a multi-channel spline field to *anything you can write as a row* — measurements,
boundary conditions, linear PDEs, quadratic PDEs — on the GPU, with the compute
kernels generated per operator at run time.

```python
from vkjet import Axes, OperatorTable, make_rows, merge_row_sets, \
                  data_operator, fit_rows, SLOT_DX, SLOT_DY, SLOT_DZ

ops = OperatorTable()                      # n_channels=5 unless declared otherwise
cid = ops.add_op("continuity", lin=[(SLOT_DX, 0, 1.0),
                                    (SLOT_DY, 1, 1.0),
                                    (SLOT_DZ, 2, 1.0)])
rows = make_rows(x_colloc, np.full(n, cid, np.int32), w, s, fuzz=0.5)

res = fit_rows(rows, ops, lo=lo, hi=hi, base_grid=(6, 6, 6, 10), n_stages=4,
               bpx=True, steps=80, stop_rel=1e-5)
u   = res.forward(x_query)          # (N, n_channels)
```

## The model

Every row is `(x, op_id, w, s, c[8], m, fuzz)`. The operator table defines, at the row's point,

```
r = c₀ + ⟨L, J(f)(x)⟩ + JᵀQJ − s          loss += ½ · scale · w · ρ_m(r)
```

with `ρ_m(r) = r²` for `m = 0` and, for `m > 0`, the wrapped Gaussian: the row asserts the
**congruence** `r ≡ 0 (mod m)` (an aliased Doppler reading with `m = 2·venc`), its loss is
periodic in `r` with the branches `k ∈ {−1, 0, +1}` marginalised at temperature `σ = τ·m`
(`fit_rows(tau=...)`; `τ = 0` the hard sawtooth, `τ < 0` the pure cosine). Under Gauss-Newton
that is soft EM against a soft-unwrapped target, so **no unwrapping precedes the solve** —
unwrapping is rows too (pair rows on the Itoh condition, authored by the caller). A row with
`fuzz = σ > 0` (cell units) is evaluated at `x + N(0, σ²)`, redrawn once per step: it declares a
measure about `x`, not a point (a fixed collocation set gets overfitted).

`J(f)(x)` is the second-order jet: **15 slots** (`val`, `∂t ∂x ∂y ∂z`, `∂tt ∂xx ∂yy ∂zz`,
`∂t∂x … ∂y∂z`) × **`n_channels`** channels. The channel count is part of the objective, not a
solver knob: it is declared on the table (`OperatorTable(n_channels=...)`, default 5) and
carried by the row file. `L` is a sparse covector on the jet, `Q` a sparse quadratic form,
`c₀` the order-0 terms of the same polynomial. An operator is registered once and referenced
by every row that uses it:

```python
ops.add_op(name,
           const=[(val, cix)],                     # c₀   (order 0)
           lin  =[(slot, ch, val, cix)],           # ⟨L, J⟩
           quad =[(s1, c1, s2, c2, val, cix)])     # JᵀQJ
```

`cix ≥ 0` multiplies that entry by the row's payload `c[cix]`, so one shared operator
serves a million different per-row covectors (beam directions, wall normals) or a
spatially varying coefficient — `ν(x)`, a graded weight — without a new kernel. The `s`
column is the order-0 term with a hard-coded minus sign; a `const` entry with `cix ≥ 0`
expresses the same per-row target through the payload, and the two agree to the bit.

**Physics are rows.** Collocation points are explicit rows and a PDE is a measurement.
λ knobs are row weights, spatially varying enforcement is a weight column, sources and
ALM shifts are the `s` column, and an experiment is a file diff (`rows_hash` gives a
canonical content hash).

**Weight folding** (`vkjet.rowauthor`). The weight is not part of the contract either: when
every coefficient of an operator rides a payload slot the residual is homogeneous in the
payload and `w·r(c)² = r(√w·c)²` exactly, so `homogenize` gives each literal-coefficient
operator a gain slot (and, on request, moves the `s` column onto a `SLOT_CONST` target
slot) and `fold` scales the payload by `√w`, leaving `w = 1`, `s = 0`. A wrapped row folds
with its modulus: its loss is homogeneous of degree 2 in `(r, m)`, so `m` scales by `√w`
too. The solver's contract is then `(x, op, payload, m, fuzz)`; "weight" and "target" are
authoring vocabulary. The folded form is not write-hostile (`reweight` scales the payload
and modulus by `√k`), but a weight is only recoverable where the operator has a dedicated
gain slot (`gain_of`); a data row's payload IS its covector and the weight is gone as a
separate quantity. Fold on the way out: the re-author diff below recognises a common
weight factor on the `w` column, not in the payload. `fit_rows` solves exactly the system it is handed — quadrature
adequacy, collocation density and coverage are the row author's responsibility, and
nothing in the solver inspects or second-guesses your rows.

### Units

Operator coefficients are in **world units**. The per-stage chain rule enters only
through `inv_widths = grid/extent`, so the jet slots are true world derivatives and
**one row file serves every grid of the dyadic ladder**.

## Execution

Semantics live in the file; execution is chosen per operator and never changes the answer.

1. **JIT** (default) — each operator's *structure* is compiled into a specialised
   kernel: entry loops unroll into literal FMAs and the gather touches only the jet
   sites that operator uses. Payload-free operators sharing an identical point set fuse
   into one shared-gather kernel. Coefficient **values** live in a small buffer, so one
   cached shader serves every value assignment of the same structure.
2. **Generic** — one table-driven kernel that can evaluate any operator. The
   always-correct floor and the semantic reference (`dispatch=False`).

Every generated kernel is **parity-verified against the generic kernel** — all four of
loss, gradient, GN diagonal and Hessian-vector product, over the full encoded domain
including the boundary cells — before first use. Any failure (no compiler, compile
error, parity miss) falls back to the generic kernel silently and correctly.

Generated SPIR-V is cached in `~/.cache/vkjet/genspv`, keyed by a hash of the **emitted
GLSL itself**, so a cache entry can never disagree with the code that would be emitted
today, and any edit to the generator invalidates it automatically.

## Optimiser

Matrix-free Gauss-Newton CG with Levenberg-Marquardt damping. `H_GN·v` comes from the
terms themselves, so the normal equations are never formed. Production path: **one cold
solve at the finest grid under the BPX multilevel preconditioner** (`bpx=True`; the ladder
grids become its levels and a scalar `steps` is the whole budget). The coarse-to-fine
ladder with a Greville-aligned warm start remains as the `bpx=False` default.

The solver carries **no implicit regulariser**: regularisation is rows (a ridge is a row,
a prior is a row), and every solver device — the CG iteration budget, the LM damping, the
BPX floor (a division guard on each level's diagonal), the step budget, the cold start —
is a convergence device whose value must not shape the answer. A truncated CG on an
unpreconditioned system *is* a Krylov regulariser; that is precisely why BPX exposed
ill-posed objectives the ladder had been hiding, and why the fix belongs in the rows.

### Convergence

`steps` is the budget. With `stop_rel > 0` the solve also stops when the objective has
**stabilised**: the mean over the last `stop_window` accepted steps of `pred/L` falls below
`stop_rel`, where `pred` is the accepted step's predicted decrease — the damped,
Krylov-truncated Newton decrement. It is affine-invariant and in the units of the
objective, so `pred/L` means the same thing across grids, row counts and clips (`‖g‖` is
neither, and not even monotone under LM). Two consecutive LM rejections at maximum damping
also stop the solve: no descent direction is left.

The threshold is checked against **one measured floor**, taken after the first accepted
step (at the cold start every physics residual is zero for every draw): the gradient is
evaluated twice at the same coefficients for the arithmetic floor (fp32 atomic order), and
with jittered rows once more at a second seed for the statistical floor of a stochastic
objective. A `stop_rel` at or below the floor is thresholding noise and is reported as such.

Stopping is a statement that the **row system is solved**, never that the answer is good: a
fit that worsens as it converges is a row problem, and stopping early would be a
regulariser in the solver. `res.diagnostics` carries `pred_rel` per accepted step,
`steps_used` and `floor`.

### Re-authoring during the solve

`rows` may be a callable `grid -> (rows, ops)`, asked once per stage and again every
`resample_every` steps (rotating collocation, weight schedules). Each answer is diffed
against the bound system **per operator**: an operator whose rows `(x, s, c, m, fuzz)` are
unchanged keeps its bound term; one whose weights changed by a single common factor keeps
its term with the factor as the term's `scale` (`w·r ≡ scale·w`); only operators whose
content actually changed are repacked and uploaded. The objective is identical to
rebinding everything. Also available: a `callback(grid, step, opt)` after every step, an
`init=(coef, grid)` warm start resized as the ladder does, and `tau`/`tau_end` for the
wrapped rows. See `PERFORMANCE.md` for recorded performance items.

## Dependencies

**volkano + numpy.** That is the whole list.

- [volkano](https://pypi.org/project/volkano/) — the Vulkan binding, installed from PyPI
  as a regular dependency.
- The B-spline basis generator (`knot_vector`, `polynomial`, `piecewise`, `bspline`)
  lives in this package, so there is no external spline dependency.

A GLSL compiler (`glslc`, or `glslangValidator` ≥ ~10) is **optional**: it enables the
JIT tier. Without one everything still runs, on the generic kernel, at roughly 5× the
cost. Discovery honours `$GLSLC` and `$GLSLANG`, then `PATH`.

```bash
pip install -e .
python3 -c "import vkjet; print(vkjet.__version__)"
```

## Tests

```bash
python3 tests/test_fit_rows.py         # end-to-end, JIT == generic
python3 tests/test_genkernel.py        # JIT parity + cache integrity
python3 tests/test_wrapped_rows.py     # congruence rows: oracle, FD, JIT parity
python3 tests/test_fuzz.py             # jittered rows: parity, determinism, sigma->0
python3 tests/test_const_term.py       # order-0 term in the table == the s column
python3 tests/test_nchannels.py        # declared channel count vs the numpy oracle
python3 tests/test_bpx_separable.py    # separable transfer == tensor product, P^T exact adjoint
python3 tests/test_rebind.py           # re-author without rebinding == rebind-all
python3 tests/test_stop.py             # stopping at stabilisation: floor, window, budget
python3 tests/test_system.py           # Field / RowSystem / Solve object layer
python3 tests/test_minibatch.py        # K=1 == full solve; norm-test batcher
python3 tests/test_rowauthor.py        # weight folding == the same objective (both tiers, wrapped rows)
```

`shaders/build.sh` rebuilds the static SPIR-V (`eqrow_*`, `kernel_apply`, `axis_csr`, the
CG/vector ops). The JIT kernels are not built here — they are generated at run time.

## Scope

vkjet is domain-free: it knows about jets, rows, bases and kernels. Application-specific
row construction — Navier-Stokes operator sets, wall/no-slip geometry, segmentation
indicator rows, unwrapping, acquisition adapters — belongs in the caller. The
cardiac-flow versions of those live in the `vkflow` reference implementation, which this
package supersedes.
