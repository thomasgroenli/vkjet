# vkjet — design record

Written for engineers maintaining or extending the solver. The README says what the
package does; this file says why it is shaped the way it is, which decisions are
deliberate, what was measured on the way, and what is still open. Dates refer to the
development log in the `vkflow` test repository, where most of these results were produced
before the code was cut into this package.

**Provenance.** The architecture is Thomas Grønli's: *physics are rows* (one least-squares
system, no privileged constraint and no privileged preprocessing), *unwrapping are rows*,
*the wall is a field*, the contract `(x, op, payload, m, fuzz)` with the weight folded into
the payload, the fuzz row field, the no-implicit-regulariser rule, and the two-layer
reading of loss versus truth. Implementations, the JIT generator, the measurements and the
mistakes recorded below are Claude's, under his direction.

---

## 1. What the package is

A domain-free least-squares solver for a multi-channel spline field on a 4D box. It knows
about jets, rows, bases and kernels. It does not know what a beam, a wall, a valve or a
Reynolds number is. Everything domain-specific is authored as rows by a caller (the cardiac
one is `cardiacrows`).

The one-sentence contract: **solve the row system it is handed, in the least-squares
sense, to convergence, and change nothing else.** Every design rule below is a
consequence of taking that sentence literally.

## 2. The contract

A row is a covector on the second-order jet of the field at a point, with a modulus and a
jitter:

```
row      = (x, op, payload c[8], m, fuzz)
residual = c₀ + ⟨L, J(f)(x)⟩ + Jᵀ Q J          (operator `op`, coefficients from c)
loss    += ½ · ρ_m(residual)      ρ_0(r) = r²,   ρ_m(r) = wrapped Gaussian, r ≡ 0 (mod m)
```

Three things that look like they belong in the contract do not:

- **The weight.** With every coefficient on a payload slot the residual is homogeneous of
  degree 1 in the payload, so `w·r(c)² = r(√w·c)²` exactly; a wrapped row's loss is
  homogeneous of degree 2 in `(r, m)`, so the modulus scales by `√w` as well. `rowauthor`
  does the fold on the way out. "Weight" is authoring vocabulary (confidence, calibration,
  λ knobs, stratified quadrature); the solver never needs it separately. What is given up:
  a data row's payload *is* its covector, so its weight is not recoverable after the fold.
- **The target.** A per-row target is the order-0 term of the same polynomial, riding a
  `SLOT_CONST` entry. The `s` column is kept for authoring convenience and agrees to the bit.
- **Any global preprocessing.** No unwrapping, no aggregation, no coverage check happens
  before the solve unless the author does it and hands over the result as rows.

The `w` and `s` columns still exist in `ROW_DTYPE` and the kernels for convenience; the
folded form is the canonical one and the hash covers both.

**The row is the K = 1 case of something more general** (derived 2026-09-14, not built):
a covector on the jet bundle discretised by quadrature, `r = Σ_p ⟨L_p, J(x_p)⟩ + Σ_pq
J(x_p)ᵀ Q_pq J(x_q) − s`. Pair rows (Itoh conditions, pressure differences, exact
finite-step transport) are K = 2; integral measurements (flow rate, sample volume) are
K = q with quadrature weights in the covector. Today's Itoh pairs approximate K = 2 by a
one-point row at the pair midpoint, which is exact only where the field is linear across
the pair. `NCPR` is a stride, not an occupancy cliff, so raising it is cheap. Revisit when
a measurement needs it.

## 3. Principles, with their reasons

### Physics are rows
Collocation points are explicit rows and a PDE is a measurement. λ knobs are row weights,
spatially varying enforcement is a weight column, a spatially varying coefficient rides
the payload, sources and ALM shifts are targets. An experiment is therefore a file, and
`rows_hash` is its identity: the same experiment hashes the same regardless of authoring
order or operator numbering; a different modulus, jitter or channel count is a different
experiment.

### Rows are the contract (2026-08-30)
The solver does not inspect the rows it is given. A coverage warning was removed because a
solver that second-guesses the author's quadrature has taken back a decision the schema
deliberately moved out, and cannot do it honestly anyway (the honest metric needs the
segmentation, which a row file does not carry). Quadrature adequacy, collocation density,
coverage and weighting are the author's job.

### Rows are representation (2026-09-03)
A row set enters the objective only through `(A, β, c)` of the quadratic it defines
(linear residuals) or the degree-4 polynomial (quadratic residuals). Row *count* is
representation, not content, so an exact re-encoding may happen anywhere, including inside
the solver, without violating the contract; only an inexact one (the ordered spatial
aggregation tier) changes the objective and stays the author's explicit call. Measured:
the per-bin eigenframe aggregation of data rows reproduces `(M, p)` to 2e-15 and compresses
232× at rank 3; bin = cell/2 costs 0.62 % of the coarse objective, bin = cell annihilates
cell-scale modes (18 %). Never truncate an eigenvalue against the bin's own λ₁: on a
fanned sector the small eigenvalues are the whole lateral observability.

### Unwrapping are rows (2026-09-11 to 09-15)
An aliased reading keeps its raw value and its modulus and asserts a congruence. Under
Gauss-Newton the wrapped Gaussian is a soft EM against a soft-unwrapped target: the GN
weight is the row's own, only the pseudo-residual changes, so the optimiser needed no
change and LM's gain ratio stays valid. The single rows' pull is bounded (periodic); pair
rows on the Itoh condition are not, so pairs push a patch across the half-wrap and the
congruence rows snap it home. Pairs alone are Ghiglia-Romero least-squares unwrapping
(non-congruent, amplitude-biased); single rows alone lock into the nearest basin.

Measured on the CFD phantom at venc 0.5: a cold BPX solve fails for *every* loss (wrapped,
annealed, cosine, bare) because the fine grid represents the sharp aliased core from step 1
and converges to the aliased basin; the ladder flips 96.7 % of branches; cold BPX plus
Itoh pair rows with a mass schedule 20 → 1 reaches 99.9 % correct branches and the MSF
reference's scores with no pre-unwrap and no ladder. In vivo the schedule was replaced by
a constant mass (the file is the experiment), and cross-frame pairs did the work the
schedule had done (see the cardiacrows record). The cosine loss matched the wrapped
Gaussian everywhere it was tried; the PINN reference's success with it came from its
smooth initialisation, not from the loss.

### No implicit regularisers (2026-09-15)
"Implicit regularizers in the solver are bad for this design. Conflates responsibilities."
Regularisation is rows. Every solver device — the CG iteration budget, the LM damping, the
BPX floor, the step budget, the cold start — is a convergence device whose value must not
shape the answer. If a fit needs damping to be acceptable, a row is missing.

Why this is a rule and not a preference: truncated CG on the old unpreconditioned system
*was* a Krylov regulariser. It read as "cg ≈ 6 gives the best u/v" for weeks. BPX removed
it and exposed objectives the truncation had been protecting, and the lateral-channel
amplification below was found only because the solver had stopped hiding it. A solver that
regularises silently makes objective mis-specification undetectable, and its protection is
resolution-dependent and unauditable.

### Loss is the solver's metric (2026-09-07)
Two layers, two metrics. The loss measures the solver; truth correlation measures the
authoring. If quality against truth degrades while the loss falls, the solver is right
and the row system's minimiser is not the truth. Corollary: **objective mis-specification
cannot be detected from inside the objective** — every statistic the solver can compute is
a function of the rows. That is why four independent stopping criteria (held-out loss,
per-coordinate gradient SNR, convergence level, an over-viscous-closure hypothesis) all
failed to reproduce the "good" early stop on the phantom: not bad luck, structurally
impossible. Do not build stopping rules to fix answers; fix rows.

The fingerprint "lower loss, worse truth" reads a whole family of earlier tuning trades as
load-bearing compensations for an under-specified objective: the BPX floor, the CG cut, the
per-stage step budget, and a quadrature bias that turned out to be doing wall
regularisation. They are a map of where rows are missing, not knobs to keep.

The practical instrument is the **row residual audit**: evaluate every operator on the
truth by finite differences, no fit. On the CFD phantom it showed the optical-flow row is
*false* (the indicator was a velocity-magnitude threshold, not a material surface) and
continuity is 25 % violated by the interpolation onto the grid. Cheap, one row at a time,
and it dissolved the CG-truncation puzzle.

### A row declares a measure, not a point (fuzz; user's idea, 2026-09-01)
At the converged production fit the physics residual was 125× larger in loss at held-out
lumen voxels than at the collocation points the fit was penalised on: a fixed collocation
set at the finest stage is overfitted (2.4× underdetermined there; the coarse stages are
overdetermined and immune). `fuzz` perturbs a row's evaluation point by N(0, σ²) in cell
units, redrawn once per optimiser step, so the objective is the mollified integral. σ is
read against the model's resolution: 0.02 cells is invisible, 0.25 already averages most
of the basis support, 0.5 is the operating plateau, 1.0 overshoots. Fuzz and rotating the
collocation set are the same requadrature seen from two sides of the compute boundary;
they do not stack, and fuzz is cheaper (no rebind). Fuzz cannot buy fewer points at the
finest stage (4× thinning with σ 0.7: overshoot, ringing, a *low* loss — the overfit
signature). Coarse-stage thinning is free.

The draw is a pure function of (row, seed) and the seed advances **once per step**: every
loss, gradient, diagonal and Hessian product inside a step must see the same points or the
line search and CG solve different problems. That is a correctness invariant, tested.

### Minibatching is the general form (2026-06 design, ported 2026-09-17)
The contract is least-squares over *minibatched* rows; the full solve is the K = 1 special
case, an alias of the same object, bit-exact on the gradient. A norm test with the
finite-population correction chooses how many buckets to draw; once the whole population is
drawn the estimator is exact and the reported noise is zero. It is a streaming form, not
a speed lever: loss and diagnostics are full-batch by design and the CG sees one operator.
Measured on real data K = 4 gave a 5× worse loss after the same steps. Gradient-SNR
stopping is structurally dead under the finite-population correction.

## 4. Execution model

Semantics live in the file; execution is chosen per operator and never changes the answer.

- **JIT by structure.** An operator's index structure (which jet sites, which channels,
  which payload slots) compiles into a specialised kernel with literal FMAs; the values
  live in a small buffer so one cached shader serves every value assignment. The
  enumeration alternative (2^2925 structures) is absurd; generation is the completion of
  the fused-kernel idea. Payload-free operators on an identical point set fuse into a
  shared-gather kernel (the hand-written NS5 shape, emitted from the table).
- **The generic kernel is the referee, not a fast path.** Every generated kernel is
  parity-checked against it on all four of loss, gradient, diagonal and HVP over the whole
  encoded domain including the boundary cells, before first use, memoised across the
  ladder. A defect confined to the last cell of one dimension once slipped through a
  check that did not cover the boundary; the check now covers it and a sabotage test
  keeps it that way.
- **Cache key ≡ emitted code.** A structure hash that stripped values while the emitter
  sorted by value once let two operators share one hash and different shaders, and the
  second silently ran the first's kernel (80 % gradient error). The key is now a hash of
  the GLSL itself. Damaged cache entries are rejected by walking the SPIR-V instruction
  stream, not by size and magic alone.
- **Kernel hints are execution metadata**, excluded from the hash, verified against the
  coefficients at dispatch; a mislabel costs speed, never correctness.
- **Recorded performance item** (`PERFORMANCE.md`): payload-carrying operators run on the
  per-operator tier one gather per row; the polar physics of the cardiac recipe carries
  six frame coefficients per point and is gathered three times instead of once. Wrapped
  operators are on their natural tier and are not an item.

## 5. Optimiser

Matrix-free Gauss-Newton CG with Levenberg-Marquardt damping. `H_GN·v` comes from the
terms; the normal equations are never formed; CG scalars stay on the device; the whole
fixed-iteration solve is recorded once into a command sequence and re-submitted (2× on
small fits, a few percent at a million records where the work is already compute-bound;
12 % of the in vivo fit once BPX was made capturable).

- **BPX cold solve retired the ladder (2026-09-05).** One cold solve at the finest grid
  under the multilevel preconditioner beats the staged ladder at matched loss, once the
  transfers were made separable and the adjoint exact. The per-axis factor was, until
  2026-09-17, a Greville-aligned linear interpolation of coefficients in index space: not
  knot insertion, off by 25 % rms on a random coarse field, and resting on three
  uniform-grid assumptions (equal Greville spacing, one refinement ratio, integer knots).
  It is now exact knot insertion by the Oslo algorithm on nested knot vectors, which is
  what separability was always licensing: the spline space is a tensor product, so the
  prolongation is a Kronecker product of 1D refinement matrices for any knot vectors.
  Measured on the in vivo recipe at the same 50 steps: loss 13313 → 12930 and the blood
  residual 0.0938 → 0.0927, same objective, same time — a convergence gain, as a
  preconditioner change must be. Non-uniform knots would still need the encode and the
  per-axis derivative scaling generalised; the transfer no longer stands in the way. The cost frontier measured then:
  a step is ~17× a CG iteration; cg 12 at 50 steps dominates cg 6 at 105; 3-stage
  semi-convergence peaks around step 35. The ladder remains as the warm start for aliased
  data and as `bpx=False`.
- **Diagonal preconditioning amplifies weakly observed channels.** On sector data in
  Cartesian channels the lateral channel is observed only through the beam angle's
  variation across a cell (1–3 % of a row), and a per-channel diagonal preconditioner
  divides its update by its own tiny curvature: the node update becomes `(r/sin a, r/cos a)`
  instead of `r·(sin a, cos a)`, off-beam and up to 100× amplified. Plain GN-CG gave a
  lateral p99 of 12.5 m/s against data at 0.58; every earlier in vivo fit's `|u|/|w| ≈ 0.6`
  was this artefact. Unpreconditioned Krylov methods and Adam never showed it because
  their updates stay in range(Aᵀ). The proposed solver fix (a per-node scalar
  preconditioner) was **rejected by the user**: the optimiser must stay blind to row
  semantics; the geometry lives in the rows. The answer became the acquisition-frame
  channels in `cardiacrows`, under which the lateral channel is exactly unobserved and there
  is nothing to amplify.
- **Exact polynomial structure.** With m = 0 and no jitter the loss is an exact quartic
  in the coefficients, so a quartic line search is exact and the central gradient
  difference is the exact Newton action at any ε (tested). GN discards < 0.4 % of the
  curvature on the phantom; Newton is available, not default.
- **Stopping at stabilisation.** `stop_rel` compares the mean predicted relative decrease
  over a window against a floor measured on the run itself: the arithmetic floor (two
  evaluations at one point, fp32 atomic order) and, for jittered rows, the statistical
  floor (a second seed) — measured after the first accepted step, because at θ = 0 the
  jitter is invisible. A threshold at or below the floor is reported as thresholding
  noise. This is a convergence statement only: a fit that worsens as it converges is a
  row problem.

## 6. Findings that shaped the design (chronological)

| when | finding | consequence |
|---|---|---|
| 2026-08 | truncated CG (cg ≈ 6) "improves" u/v on z-only data | it was Krylov regularisation of an under-specified objective; retired with BPX |
| 2026-08-20 | u/v ceiling on the phantom is observation geometry, not the optimiser | lateral channels are inferred through physics; wall Cauchy data is the lever |
| 2026-08-28 | equations as rows ≡ hand-written fused kernels to 2e-7; JIT ≈ hand-written speed | the operator table replaced the fused tiers |
| 2026-09-01 | 125× physics overfit at fixed collocation | fuzz row field |
| 2026-09-03 | quadrature fidelity is not quality (fixing a weight bias hurt) | confirm every fidelity win with a scored run |
| 2026-09-05 | BPX cold solve at matched loss beats the ladder | ladder retired; `steps` is the whole budget |
| 2026-09-07 | four stopping criteria all fail; loss is the solver's metric | no truth-driven stopping rules |
| 2026-09-08 | optical flow is false on the CFD phantom; continuity 25 % violated | row residual audit before fitting |
| 2026-09-11 | cold BPX locks into the aliased basin for every wrap loss | coarse-to-fine or pair rows are mandatory on aliased data |
| 2026-09-14 | Itoh pair rows unwrap the phantom; mass schedule sidesteps the ladder | unwrapping stays inside the contract |
| 2026-09-14 | in vivo "improvement" was two failed fits ranked by proxies | basics first; judge on video and wrapped-back panels |
| 2026-09-15 | lateral amplification by the diagonal preconditioner, mechanism confirmed bit-exactly | acquisition-frame channels; no geometry in the solver |
| 2026-09-15 | `make_rows` silently dropped `fuzz`; all "fuzz" runs were unjittered | regression test; retracted claims |
| 2026-09-17 | folding scales the modulus; K = 1 minibatch is an exact alias | contract as stated in §2 |
| 2026-09-17 | the BPX transfer was linear interpolation, 25 % off on a rough field | exact knot insertion (Oslo); loss −2.9 % at a fixed budget |

## 7. Open items

- **Per-step jitter under LM.** The gain ratio compares a prediction and a measurement
  taken on two different draws, reads noise (pred/L flat at ~0.6, floor 1.2e-2), and the
  solve crawls. Options: block-fixed draws over a few steps, or a sample average per step.
  User's call; fixed budgets are used meanwhile.
- **The fixed step budget is an implicit regulariser** on the lateral channel and on the
  residual rows: spurious flow grows with convergence. The stopping rule found it; the
  rows have not answered it.
- **The statistical floor exists and no production run has used it.**
- **Deflation** of the null space was the one validated tool of the vorticity work; not
  ported.
- **Payload on the grouped tier** (`PERFORMANCE.md`).
- Done 2026-09-17: the captured CG solve now covers BPX (per-level metas made static);
  on the in vivo recipe the fit went 40.5 → 35.8 s, i.e. about 17 % of the optimisation
  stage, the rest being compute-bound as the earlier measurement predicted.
- **Multi-point rows** (§2).

## 8. Testing doctrine

Tests ship with the package (`python -m vkjet.tests`), plain unittest, one shared
context, none over ten seconds. The rules they encode:

- The generic kernel is checked against a float64 numpy oracle on every slot class; every
  JIT kernel is checked against the generic one on all four kernels including boundary
  cells; the oracle's identities are per row, so a small row set suffices.
- Too-exact agreement is a bug signal (a configuration knob that prints "on" while
  running the baseline was found this way twice).
- Every invariant that once broke has a test: the cache key, the boundary cells, the
  seed-per-step rule, the loss main having its own prelude, `make_rows` carrying every
  column, K = 1 being an alias, the modulus scaling under the fold.
- Determinism is a floor, not an equality: fp32 atomics give ~1e-7 between repeated
  evaluations; a repeated seed must sit at that floor and a changed seed must move by
  orders of magnitude more.
