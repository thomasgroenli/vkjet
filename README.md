# vkjet

Fit a multi-channel spline field to *anything you can write as a row* — measurements,
boundary conditions, linear PDEs, quadratic PDEs — on the GPU, with the compute
kernels generated per operator at run time.

```python
from vkjet import Axes, OperatorTable, make_rows, merge_row_sets, \
                  data_operator, fit_rows, SLOT_DX, SLOT_DY, SLOT_DZ

ops = OperatorTable()
cid = ops.add_op("continuity", lin=[(SLOT_DX, 0, 1.0),
                                    (SLOT_DY, 1, 1.0),
                                    (SLOT_DZ, 2, 1.0)])
rows = make_rows(x_colloc, np.full(n, cid, np.int32), w, s)

res = fit_rows(rows, ops, lo=lo, hi=hi, base_grid=(6, 6, 6, 10), n_stages=4)
u   = res.forward(x_query)          # (N, 5)
```

## The model

Every row is `(x, op_id, w, s, c[8], m, fuzz)`. The operator table defines, at the row's point,

```
r = ⟨L, J(f)(x)⟩ + JᵀQJ − s          loss += ½ · scale · w · ρ_m(r)
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
`∂t∂x … ∂y∂z`) × **5 channels**. `L` is a sparse covector on that jet, `Q` a sparse
quadratic form. An operator is registered once and referenced by every row that uses it:

```python
ops.add_op(name,
           lin =[(slot, ch, val, cix)],            # ⟨L, J⟩
           quad=[(s1, c1, s2, c2, val, cix)])      # JᵀQJ
```

`cix ≥ 0` multiplies that entry by the row's payload `c[cix]`, so one shared operator
serves a million different per-row covectors (beam directions, wall normals) or a
spatially varying coefficient — `ν(x)`, a graded weight — without a new kernel.

**Physics are rows.** Collocation points are explicit rows and a PDE is a measurement.
λ knobs are row weights, spatially varying enforcement is a weight column, sources and
ALM shifts are the `s` column, and an experiment is a file diff (`rows_hash` gives a
canonical content hash). `fit_rows` solves exactly the system it is handed — quadrature
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
grids become its levels). The coarse-to-fine ladder with a Greville-aligned warm start
remains as the `bpx=False` fallback.

The solver carries **no implicit regulariser**: regularisation is rows (a ridge is a row,
a prior is a row), and every solver device — the CG iteration budget, the LM damping, the
BPX floor (a division guard on each level's diagonal), the step budget, the cold start —
is a convergence device whose value must not shape the answer. A truncated CG on an
unpreconditioned system *is* a Krylov regulariser; that is precisely why BPX exposed
ill-posed objectives the ladder had been hiding, and why the fix belongs in the rows.
Convergence is a fixed step budget for now; stopping at stabilisation is the open item.

Callers can supply `rows` as a callable `grid -> (rows, ops)` re-asked every
`resample_every` steps (rotating collocation, mass schedules), a `callback(grid, step,
opt)`, an `init=(coef, grid)` warm start, and `tau`/`tau_end` for the wrapped rows.
See `PERFORMANCE.md` for recorded performance items.

## Dependencies

**volkano + numpy.** That is the whole list.

- [volkano](https://github.com/thomgronli/volkano) — the Vulkan binding. Not on PyPI;
  clone it and put it on `PYTHONPATH`.
- The B-spline basis generator is **vendored** in `vkjet/_gs` (from `genspline`), so
  there is no external spline dependency. `tests/test_bspline_vendor.py` cross-checks
  the vendored copy against upstream bit-for-bit whenever upstream is importable.

A GLSL compiler (`glslc`, or `glslangValidator` ≥ ~10) is **optional**: it enables the
JIT tier. Without one everything still runs, on the generic kernel, at roughly 5× the
cost. Discovery honours `$GLSLC` and `$GLSLANG`, then `PATH`.

```bash
git clone <volkano> ~/projects/volkano
PYTHONPATH=~/projects/volkano python3 -c "import vkjet; print(vkjet.__version__)"
```

## Tests

```bash
PYTHONPATH=~/projects/volkano python3 tests/test_bspline_vendor.py   # vendor == upstream
PYTHONPATH=~/projects/volkano python3 tests/test_fit_rows.py         # end-to-end, JIT == generic
PYTHONPATH=~/projects/volkano python3 tests/test_genkernel.py        # JIT parity + cache integrity
PYTHONPATH=~/projects/volkano python3 tests/test_wrapped_rows.py     # congruence rows: oracle, FD, JIT parity
PYTHONPATH=~/projects/volkano python3 tests/test_fuzz.py             # jittered rows: parity, determinism, sigma->0
```

`shaders/build.sh` rebuilds the static SPIR-V (`eqrow_*`, `kernel_apply`, the CG/vector
ops). The JIT kernels are not built here — they are generated at run time.

## Scope

vkjet is domain-free: it knows about jets, rows, bases and kernels. Application-specific
row construction — Navier-Stokes operator sets, wall/no-slip geometry, segmentation
indicator rows, unwrapping, acquisition adapters — belongs in the caller. The
cardiac-flow versions of those live in the `vkflow` reference implementation, which this
package supersedes.
