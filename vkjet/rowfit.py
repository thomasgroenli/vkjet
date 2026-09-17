"""fit_rows — solve a jet-row system on a dyadic spline ladder.

A row file is the complete specification of the objective. Every row is
(x, op_id, w, s, c) and the operator table defines

    r = <L, J(f)(x)> + J^T Q J - s ,     loss += 1/2 * scale * w * r^2

over the second-order jet of an NCH-channel field. Data rows, boundary
conditions, linear PDEs and quadratic PDEs are all the same thing: collocation
points are explicit rows, and any (<= quadratic) PDE is a measurement.

    res = fit_rows("everything.npz", lo=lo, hi=hi)
    u   = res.forward(x_query)

PHYSICS ARE ROWS. fit_rows solves exactly the row system it is handed. Whether
that system is well posed — collocation density, coverage of the finest grid,
weighting, quadrature adequacy — is the ROW AUTHOR's responsibility. Nothing
here inspects, second-guesses or warns about the supplied rows.

Execution is an implementation detail chosen per operator, never a semantic
one. Two tiers:

  1. JIT (default) — each operator's structure is compiled into a specialised
     kernel (vkjet.genkernel); payload-free operators sharing an identical
     point set fuse into ONE shared-gather kernel. Every generated kernel is
     parity-verified against the generic kernel before first use, and any
     failure (no toolchain, compile error, parity miss) falls back silently and
     correctly.
  2. Generic EqRowTerm — the always-correct floor and the semantic reference.
     dispatch=False forces it for everything.

Operator coefficients are in WORLD units. The per-stage chain rule enters only
through inv_widths = grid/extent, so ONE row file serves every grid of the
ladder.
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Sequence

import numpy as np

from .apply import ApplyForward
from .context import Context
from .data import Axes, load_rows
from .eqrow import OperatorTable, EqRowTerm, NCH
from .optim import GaussNewtonCG, _MaxReduce


# --------------------------------------------------------------------------- #
# coarse -> fine warm start                                                   #
# --------------------------------------------------------------------------- #
def refine_matrix(n_in: int, n_out: int, order: int = 4) -> np.ndarray:
    """(n_out x n_in) EXACT prolongation between two periodic uniform B-spline
    bases whose knot vectors are nested (n_out a multiple of n_in), by the Oslo
    algorithm (discrete B-splines, Cohen-Lyche-Riesenfeld):

        B^coarse_j = sum_i alpha_j(i) B^fine_i,   P[i, j] = alpha_j(i)

    with alpha built by the de Boor-like recursion over the fine knots
        alpha_{j,1}(i) = [tau_j <= t_i < tau_{j+1}]
        alpha_{j,k}(i) = (t_{i+k-1} - tau_j)/(tau_{j+k-1} - tau_j) alpha_{j,k-1}(i)
                       + (tau_{j+k} - t_{i+k-1})/(tau_{j+k} - tau_{j+1}) alpha_{j+1,k-1}(i).
    Knot-vector B-spline m (support [t_m, t_{m+order}]) is the kernel's
    coefficient m + order - 1 (LOOKBACK layout), wrapped modulo the extent.
    Exact to float precision (the coarse field is reproduced), where the
    Greville-interpolation transfer it replaces was off by 25% rms on a random
    coarse field. Separable: one such factor per axis. Non-nested pairs
    (`n_out % n_in != 0`) have no exact embedding and raise."""
    if n_in == 1:
        return np.ones((n_out, 1), dtype=np.float64)
    if n_out % n_in:
        raise ValueError(f"knot vectors not nested: {n_in} -> {n_out}")
    if n_out == n_in:
        return np.eye(n_out)
    r = n_out // n_in
    ext = order + 1                                  # periodic extension on each side
    tau = (np.arange(-ext, n_in + ext + 1) * r).astype(np.float64)    # coarse knots in fine units
    t = np.arange(-ext * r, n_out + ext * r + 1).astype(np.float64)   # fine knots
    off_c, off_f = ext, ext * r                      # index of knot 0 in each array
    P = np.zeros((n_out, n_in))
    for j0 in range(n_in):                           # one period of coarse functions
        j = j0 + off_c
        # fine functions overlapping the coarse support [tau_j, tau_{j+order}]
        i_lo = int(tau[j]) - order + off_f + 1 + (-1)
        i_hi = int(tau[j + order]) + off_f
        for i in range(max(i_lo, 0), min(i_hi, len(t) - order - 1)):
            # alpha_{jj, k}(i) for jj = j .. j+order-1, k = 1 .. order
            a = np.array([1.0 if tau[jj] <= t[i] < tau[jj + 1] else 0.0
                          for jj in range(j, j + order)])
            for k in range(2, order + 1):
                nxt = np.zeros(order - k + 1)
                for m in range(order - k + 1):
                    jj = j + m
                    d1 = tau[jj + k - 1] - tau[jj]; d2 = tau[jj + k] - tau[jj + 1]
                    nxt[m] = ((t[i + k - 1] - tau[jj]) / d1 * a[m] if d1 > 0 else 0.0) \
                        + ((tau[jj + k] - t[i + k - 1]) / d2 * a[m + 1] if d2 > 0 else 0.0)
                a = nxt
            if a[0] != 0.0:
                P[(i - off_f + order - 1) % n_out, (j0 + order - 1) % n_in] += a[0]
    return P


def _axis_resize_matrix(n_in: int, n_out: int, order: int = 4) -> np.ndarray:
    """(n_out x n_in) per-axis transfer of a coefficient grid: the exact
    knot-insertion embedding when the grids nest (the dyadic ladder, the BPX
    levels), otherwise a Greville-aligned periodic linear interpolation of the
    coefficients — a warm-start heuristic only (control point k of a cubic sits
    at Greville abscissa k - 1 in this layout). Identity when n_in == n_out."""
    if n_in == 1 or n_out % n_in == 0:
        return refine_matrix(n_in, n_out, order)
    g = order - 1
    pos = g + (np.arange(n_out) - g) * (n_in / n_out)
    lo = np.floor(pos).astype(int)
    frac = pos - lo
    M = np.zeros((n_out, n_in), dtype=np.float64)
    rows = np.arange(n_out)
    M[rows, lo % n_in] += 1.0 - frac
    M[rows, (lo + 1) % n_in] += frac
    return M


def multilinear_resize(coef_coarse, ext_coarse, ext_fine, n_channels,
                       order: int = 4) -> np.ndarray:
    """Tensor-product Greville-aligned periodic resize of a coef grid."""
    arr = np.ascontiguousarray(coef_coarse, np.float64).reshape(
        tuple(ext_coarse) + (n_channels,))
    for ax in range(len(ext_coarse)):
        M = _axis_resize_matrix(ext_coarse[ax], ext_fine[ax], order)
        arr = np.tensordot(M, arr, axes=([1], [ax]))
        arr = np.moveaxis(arr, 0, ax)
    return arr.astype(np.float32).reshape(-1)


class FitResult:
    """Fitted coefficients + basis metadata + a forward evaluator."""

    def __init__(self, ctx, coef, axes, bases, extents, stage_losses,
                 n_channels=NCH, diagnostics=None):
        self.ctx = ctx
        self.coef = coef
        self.axes = axes
        self.bases = bases
        self.extents = extents
        self.stage_losses = stage_losses
        self.n_channels = n_channels
        self.diagnostics = diagnostics or {}
        self._af = ApplyForward(ctx, bases, n_channels)

    def forward(self, x_world) -> np.ndarray:
        """(N, D) world coordinates -> (N, n_channels) field values."""
        return self._af.forward(
            self.axes.encode(np.asarray(x_world, np.float32)),
            self.coef.reshape(self.extents + (self.n_channels,)))


# --------------------------------------------------------------------------- #
# execution-tier selection (semantics-preserving)                             #
# --------------------------------------------------------------------------- #
def _jit_terms(ctx, axes, bases, grid, x, op, w, s, c, ops, iw, verbose,
               fuzz=None, modulus=None, of_row=None, nb=1):
    """Compile a specialised kernel per operator where possible.

    Payload-free operators that share an IDENTICAL point set fuse into one
    shared-gather kernel; the rest get a per-operator kernel. Returns
    (terms, handled_mask) with terms as (term, frozenset of op ids) pairs;
    anything not handled is left to the generic term.

    Wrapped rows (modulus m > 0: r = 0 mod m) and jittered rows (fuzz > 0)
    ride the per-row record. The grouped kernel carries no modulus, so an
    operator with any wrapped row stays off it (silently treating a congruence
    as an equality would change the objective); it carries ONE sigma per point,
    so co-located operators must agree on sigma or they go per-op.

    `of_row` (global per-row bucket id, nb buckets) binds every term with the
    SAME partition of the one row system (minibatching); None = one batch.
    """
    from .rowbatch import local_buckets
    lb = ((lambda sel: [np.arange(len(sel))]) if of_row is None
          else (lambda sel: local_buckets(of_row, sel, nb)))
    handled = np.zeros(len(x), bool)
    terms = []
    try:
        from .genkernel import (GeneratedRowTerm, GroupedRowTerm,
                                verify_generated, verify_grouped,
                                find_compiler)
        comp = find_compiler()
    except Exception:
        comp = None
    if comp is None:
        if verbose:
            print("    [dispatch] no glslc/glslangValidator — generic only",
                  flush=True)
        return terms, handled
    fz = None if fuzz is None else np.ascontiguousarray(fuzz, np.float32).reshape(-1)
    mm = None if modulus is None else np.ascontiguousarray(modulus, np.float32).reshape(-1)
    wrap_ops = set() if mm is None else {int(k) for k in np.unique(op[mm > 0])}

    # --- grouped: payload-free ops on a shared point set -------------------- #
    payload_free = [
        int(k) for k in np.unique(op)
        if all(e[3] < 0 for e in ops.lin[k]) and all(q[5] < 0 for q in ops.quad[k])
        and k not in wrap_ops]
    sig = {}
    for k in payload_free:
        rows = np.flatnonzero(op == k)
        srt = rows[np.lexsort(x[rows].T)]
        sig.setdefault(hashlib.sha256(x[srt].tobytes()).hexdigest(),
                       []).append((k, srt))
    for members in sig.values():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda t: t[0])
        gids = [k for k, _ in members]
        sg = None
        if fz is not None and any((fz[srt] > 0).any() for _, srt in members):
            cols = [fz[srt] for _, srt in members]
            if not all(np.array_equal(cols[0], cc) for cc in cols[1:]):
                if verbose:
                    print(f"    [dispatch] jit-group {[ops.names[k] for k in gids]}: "
                          f"sigma differs between co-located rows — per-op",
                          flush=True)
                continue
            sg = cols[0]
        try:
            gt = GroupedRowTerm(ctx, bases, iw, ops, gids, compiler=comp)
            verify_grouped(ctx, gt, bases, iw, ops, gids)
            xp = x[members[0][1]]
            W = np.stack([w[r] for _, r in members], 1)
            S = np.stack([s[r] for _, r in members], 1)
            if of_row is None:
                gt.bind_points(axes.encode(xp), W, S, sigma=sg)
            else:
                gt.bind_point_buckets(axes.encode(xp), W, S, lb(members[0][1]), sigma=sg)
            terms.append((gt, frozenset(gids)))
            for _, r in members:
                handled[r] = True
            if verbose:
                print(f"    [dispatch] jit-group "
                      f"{[ops.names[k] for k in gids]} @ {len(xp):,} pts",
                      flush=True)
        except Exception as ex:
            if verbose:
                print(f"    [dispatch] jit-group failed ({type(ex).__name__}: "
                      f"{ex}) — per-op", flush=True)

    # --- per-operator ------------------------------------------------------- #
    for k in np.unique(op[~handled]):
        m = (op == k) & ~handled
        try:
            gt = GeneratedRowTerm(ctx, bases, iw, ops, int(k), compiler=comp)
            verify_generated(ctx, gt, bases, iw, ops, int(k))
            if of_row is None:
                gt.bind_batch(axes.encode(x[m]), np.zeros(int(m.sum()), np.int32),
                              w[m], s[m], c[m],
                              fuzz=(None if fz is None else fz[m]),
                              modulus=(None if mm is None else mm[m]))
            else:
                gt.bind_buckets(axes.encode(x[m]), np.zeros(int(m.sum()), np.int32),
                                w[m], s[m], c[m], lb(np.flatnonzero(m)),
                                fuzz=(None if fz is None else fz[m]),
                                modulus=(None if mm is None else mm[m]))
            terms.append((gt, frozenset([int(k)])))
            handled |= m
        except Exception as ex:
            if verbose:
                print(f"    [dispatch] jit '{ops.names[int(k)]}' failed "
                      f"({type(ex).__name__}: {ex}) — generic", flush=True)
    if verbose and wrap_ops:
        print(f"    [dispatch] wrapped ops {sorted(wrap_ops)} "
              f"({int((mm > 0).sum()):,} rows with m > 0) on the per-row "
              f"record path", flush=True)
    return terms, handled


# --------------------------------------------------------------------------- #
def multilinear_restrict(coef_fine: np.ndarray, ext_fine: Sequence[int],
                         ext_coarse: Sequence[int], n_channels: int,
                         order: int = 4) -> np.ndarray:
    """Exact ADJOINT (transpose) of multilinear_resize: fine → coarse via Pᵀ.

    Restriction must be the transpose of prolongation (same axis matrices, transposed)
    — NOT an independent fine→coarse resize — so that the BPX preconditioner
    C⁻¹ = Σ_l P_l D_l⁻¹ P_lᵀ is symmetric. Used by the multilevel preconditioner.
    """
    nd = len(ext_fine)
    arr = np.ascontiguousarray(coef_fine, np.float64).reshape(tuple(ext_fine) + (n_channels,))
    for ax in range(nd):
        M = _axis_resize_matrix(ext_coarse[ax], ext_fine[ax], order)   # (fine, coarse)
        arr = np.tensordot(M.T, arr, axes=([1], [ax]))   # contract fine → coarse
        arr = np.moveaxis(arr, 0, ax)
    return arr.astype(np.float32).reshape(-1)


def fit_rows(rows, ops=None, lo=None, hi=None, base_grid=(6, 6, 6, 12),
             n_stages=3, steps=None, cg_iters=12, n_channels=NCH,
             periodic=(True, False, False, False), dispatch=True,
             resample_every=0, bpx=False, bpx_floor=1e-2,
             tau=0.0, tau_end=None, init=None, callback=None, seed=0,
             stop_rel=0.0, stop_window=5, minibatch=0, batch_tol=0.15,
             ctx=None, verbose=True):
    """Fit a spline field from jet rows (array pair, or a save_rows path).

    `rows` may also be a CALLABLE grid -> (rows, ops): it is asked once per
    stage for the rows appropriate to that grid, and with ``resample_every=k``
    again every k steps within a stage, so an author can rotate a collocation
    set (or re-author any row family) during the solve. fit_rows solves
    exactly what it is handed each time.

    dispatch=True  JIT-generated kernels, generic for anything they cannot take
                   ("jit" is accepted as a synonym)
    dispatch=False pure generic EqRowTerm (the semantic reference)

    BPX (bpx=True): the ladder grids become preconditioner LEVELS and the
    solve is ONE cold run at the finest of them; a scalar `steps` is then the
    total budget. `bpx_floor` is each level's damping relative to its own max
    diagonal — a division guard, not a regulariser (regularisation is rows).

    WRAPPED ROWS: a row whose `nyquist` column is m > 0 asserts the congruence
    r = 0 (mod m). Its loss is the wrapped Gaussian (branches k in {-1,0,1} at
    temperature sigma = tau*m); under GN it is soft EM, so no unwrapping
    precedes the solve. `tau` is constant unless `tau_end` is given (linear
    anneal over each stage's steps); tau = 0 is the hard sawtooth, tau < 0 the
    pure cosine. Rows with m = 0 are the plain equality.

    JITTERED ROWS: a row with `fuzz` = sigma > 0 (cell units) is evaluated at
    x + N(0, sigma^2), redrawn once per step (one draw shared by every dispatch
    of that step).

    `init` = (coef, grid): warm-start the first stage from a field on another
    grid (resized as the ladder does between stages). `callback(grid, step,
    opt)` runs after every step.

    CONVERGENCE. `steps` is the budget. With ``stop_rel > 0`` the solve also
    stops when the objective has STABILISED: the mean over the last
    `stop_window` accepted steps of pred/L <= stop_rel, where pred is the
    accepted step's predicted decrease — the damped, Krylov-truncated Newton
    decrement, affine-invariant and in the units of the objective, so pred/L
    means the same thing across grids, row counts and clips (||g|| is neither,
    and not even monotone under LM). It is a statement that the ROW SYSTEM IS
    SOLVED, never that the answer is good: a fit that worsens as it converges
    is a row problem, and stopping early would be a regulariser in the solver.
    ONE FLOOR, MEASURED: before the solve the gradient is evaluated twice at
    the same coefficients; their disagreement is the arithmetic floor (fp32
    atomic order, ~3e-7). With jittered rows it is evaluated again with two
    seeds, which is the statistical floor of a stochastic objective (the same
    mechanism a minibatched solve will use). stop_rel below the floor is
    thresholding noise and is reported as such. A second condition is free:
    two consecutive LM rejections at maximum damping mean no descent
    direction is left. The diagnostics carry `pred_rel` per accepted step,
    `steps_used` and `floor`.

    MINIBATCHING (minibatch > 0 = target rows per bucket): the row system is
    partitioned into K = len(rows)//minibatch buckets and each outer step
    draws buckets in random order until the norm test passes (`batch_tol`),
    so the effective batch self-calibrates. This is the GENERAL form of the
    contract, not an optimisation of it: the full solve is K=1, where the
    bucket IS the full row set (aliased, no copy) and the dispatches are
    those of minibatch=0 (tests/test_minibatch.py). Every term is bucketed —
    no deterministic baseline, since that would mean the solver deciding
    which rows are data and which are physics. diag and loss stay full-batch:
    diag is the preconditioner, loss is the real objective that LM
    accept/reject and the stop test read. The stopping floor then measures
    itself in the right regime: two gradients at the same coefficients differ
    by fp32 order at K=1 and by the bucket draw at K>1.

    Returns a :class:`FitResult`.
    """
    t0 = time.time()
    from .rowbatch import row_buckets, BatchedRowSystem
    per_stage = callable(rows)
    if isinstance(rows, (str, os.PathLike)):
        rows, ops = load_rows(rows)
    assert lo is not None and hi is not None
    assert per_stage or ops is not None
    if ops is not None:
        n_channels = int(getattr(ops, "n_channels", n_channels))
    own_ctx = ctx is None
    ctx = ctx or Context()
    if dispatch == "jit":
        dispatch = True

    grids = [tuple(g * (1 << k) if g > 1 else 1 for g in base_grid)
             for k in range(n_stages)]
    if steps is None:
        steps = (25, 25, 30, 25, 25)[:n_stages]
    lv_grids = ()
    if bpx:
        lv_grids = tuple(grids)
        grids = [grids[-1]]
        steps = ((int(steps),) if np.isscalar(steps)
                 else (int(sum(steps)),))
    if np.isscalar(steps):
        steps = (int(steps),) * n_stages
    ext = np.asarray(hi, np.float64) - np.asarray(lo, np.float64)
    nd = len(ext)

    def cell_ids(pts, grid):
        ax = Axes(lo, hi, grid, periodic=periodic)
        e = np.floor(ax.encode(pts)).astype(np.int64)
        cid = np.zeros(len(pts), np.int64)
        for k, g in enumerate(grid):
            cid = cid * g + np.clip(e[:, k], 0, g - 1)
        return cid

    def unpack(rws, ops_, sort_grid):
        """Row arrays + one stable (cell, op) sort: scatter locality plus
        warp-uniform operator ids inside a cell."""
        names = rws.dtype.names or ()
        xx = np.ascontiguousarray(rws["x"], np.float32)
        oo = np.ascontiguousarray(rws["op"], np.int32)
        ww = np.ascontiguousarray(rws["w"], np.float32)
        ss = np.ascontiguousarray(rws["s"], np.float32)
        cc = np.ascontiguousarray(rws["c"], np.float32)
        ff = (np.ascontiguousarray(rws["fuzz"], np.float32) if "fuzz" in names
              else np.zeros(len(rws), np.float32))
        mq = (np.ascontiguousarray(rws["nyquist"], np.float32) if "nyquist" in names
              else np.zeros(len(rws), np.float32))
        key = cell_ids(xx, sort_grid) * (ops_.n_ops + 1) + oo
        order = np.argsort(key, kind="stable")
        xx, oo, ww, ss, cc, ff, mq = (a[order] for a in (xx, oo, ww, ss, cc, ff, mq))
        return np.ascontiguousarray(xx), oo, ww, ss, cc, ff, mq, order

    def make_terms(ax_, bs_, g, xx, oo, ww, ss, cc, ff, mq, ops_, verbose_, of_row=None, nb=1):
        """Terms for one row set on grid g (JIT where possible, generic rest);
        with `of_row` every term is bound with the same bucket partition."""
        from .rowbatch import local_buckets
        iw_ = [g[k] / ext[k] for k in range(nd)]
        if dispatch:
            tt, hd = _jit_terms(ctx, ax_, bs_, g, xx, oo, ww, ss, cc, ops_, iw_,
                                verbose_, fuzz=ff, modulus=mq, of_row=of_row, nb=nb)
            if not hd.all():
                gen = EqRowTerm(ctx, bs_, iw_, ops_)
                r = np.flatnonzero(~hd)
                if of_row is None:
                    gen.bind_batch(ax_.encode(xx[r]), oo[r], ww[r], ss[r], cc[r],
                                   fuzz=ff[r], modulus=mq[r])
                else:
                    gen.bind_buckets(ax_.encode(xx[r]), oo[r], ww[r], ss[r], cc[r],
                                     local_buckets(of_row, r, nb), fuzz=ff[r], modulus=mq[r])
                tt.append((gen, frozenset(int(k) for k in np.unique(oo[r]))))
            if verbose_:
                print(f"    [dispatch] jit {int(hd.sum()):,}  "
                      f"generic {int((~hd).sum()):,}", flush=True)
        else:
            t_ = EqRowTerm(ctx, bs_, iw_, ops_)
            if of_row is None:
                t_.bind_batch(ax_.encode(xx), oo, ww, ss, cc, fuzz=ff, modulus=mq)
            else:
                t_.bind_buckets(ax_.encode(xx), oo, ww, ss, cc, row_buckets_from(of_row, nb),
                                fuzz=ff, modulus=mq)
            tt = [(t_, frozenset(int(k) for k in np.unique(oo)))]
        return tt

    def row_buckets_from(of_row, nb):
        return [np.flatnonzero(of_row == i) for i in range(nb)]

    if not per_stage:
        x, op, w, s, c, fz, mq, row_order = unpack(rows, ops, grids[-1])
    coef, prev = None, None
    if init is not None:
        coef = np.ascontiguousarray(init[0], np.float32).reshape(-1)
        prev = tuple(int(g) for g in init[1])
        assert coef.size == int(np.prod(prev)) * n_channels, (coef.size, prev)
        if verbose:
            print(f"  [init] warm start from a {prev} field", flush=True)
    losses = []; eff_hist = []
    for si, (grid, n_steps) in enumerate(zip(grids, steps)):
        if per_stage:
            stage_rows, ops = rows(grid)
            n_channels = int(getattr(ops, "n_channels", n_channels))
            x, op, w, s, c, fz, mq, row_order = unpack(stage_rows, ops, grid)
            if verbose:
                print(f"  [rows] {len(x):,} rows for {grid}", flush=True)
        axes = Axes(lo, hi, grid, periodic=periodic)
        bases = axes.bases()
        extents = tuple(b.primal_extent for b in bases)
        n = int(np.prod(grid)) * n_channels
        coef = (np.zeros(n, np.float32) if coef is None
                else multilinear_resize(coef, prev, grid, n_channels))
        opt = GaussNewtonCG(ctx, n); opt.set_coef(coef)
        _tau_now = [float(tau)]
        _first = [True]
        bound = {}          # opset -> dict(term, keys{op: content key}, w0{op: w}, scale)

        _draw = [0]

        def build_terms():
            """Terms for the CURRENT rows. On a re-author, an operator whose
            rows (x, s, payload, m, fuzz) are unchanged keeps its bound term;
            if only its weights changed by one common factor the factor
            becomes the term's scale (w·r ≡ scale·w — the objective is the
            same, no rebind); anything else is rebound. Rows stay the whole
            specification; this is representation, not content.
            With minibatching every term is bound with one fresh bucket
            partition and the whole system is wrapped as ONE stochastic term."""
            if minibatch:
                nb = max(1, len(x) // int(minibatch))
                of_row = np.empty(len(x), np.int32)
                for _i, _ix in enumerate(row_buckets(len(x), nb, seed=seed + 7919 * si + _draw[0])):
                    of_row[_ix] = _i
                _draw[0] += 1
                tt = make_terms(axes, bases, grid, x, op, w, s, c, fz, mq, ops,
                                verbose and si == 0 and _first[0], of_row=of_row, nb=nb)
                for t_, _ in tt:
                    if hasattr(t_, "set_tau"):
                        t_.set_tau(_tau_now[0])
                sysw = BatchedRowSystem(ctx, [t_ for t_, _ in tt], n, tol=batch_tol,
                                        seed=seed + 104729 * si)
                if verbose and _first[0]:
                    print(f"  [batch] {len(x):,} rows -> {nb} bucket(s) of ~{len(x) // nb:,}, "
                          f"norm-test tol={batch_tol:g}{'  (K=1: identity)' if nb == 1 else ''}",
                          flush=True)
                _first[0] = False
                return [sysw]
            first = not bound
            keys = {}; wnow = {}
            for k in np.unique(op):
                sel = op == k
                h = hashlib.blake2b(digest_size=16)
                for a in (x[sel], s[sel], c[sel], mq[sel], fz[sel]):
                    h.update(np.ascontiguousarray(a).tobytes())
                keys[int(k)] = h.digest(); wnow[int(k)] = w[sel]
            keep, changed = [], set(keys)
            n_kept = n_scaled = 0
            for opset, rec in list(bound.items()):
                if not opset <= set(keys) or any(keys[k] != rec["keys"][k] for k in opset):
                    continue
                ratios = []
                for k in opset:
                    w0, w1 = rec["w0"][k], wnow[k]
                    if w0.shape != w1.shape:
                        ratios = None; break
                    if np.array_equal(w0, w1):
                        ratios.append(1.0); continue
                    r = float(np.median(w1[w0 > 0] / w0[w0 > 0])) if (w0 > 0).any() else None
                    if r is None or not np.allclose(w1, r * w0, rtol=2e-6, atol=0.0):
                        ratios = None; break
                    ratios.append(r)
                if ratios is None or not np.allclose(ratios, ratios[0], rtol=1e-6):
                    continue
                r = ratios[0]
                if r != rec["scale"]:
                    if not hasattr(rec["term"], "set_scale") or len(opset) > 1:
                        continue          # no scale interface (grouped): rebind
                    rec["term"].set_scale(r); rec["scale"] = r; n_scaled += 1
                keep.append(opset); changed -= opset; n_kept += 1
            for opset in list(bound):
                if opset not in keep:
                    del bound[opset]
            new = []
            if changed:
                sel = np.isin(op, sorted(changed))
                new = make_terms(axes, bases, grid, x[sel], op[sel], w[sel], s[sel], c[sel],
                                 fz[sel], mq[sel], ops, verbose and si == 0 and _first[0])
                for t_, opset in new:
                    if hasattr(t_, "set_tau"):
                        t_.set_tau(_tau_now[0])
                    bound[opset] = dict(term=t_, keys={k: keys[k] for k in opset},
                                        w0={k: wnow[k].copy() for k in opset}, scale=1.0)
            if verbose and not first:
                n_re = int(sum(len(bound[o]["w0"][k]) for _, o in new for k in o))
                print(f"    [rebind] {n_kept} term(s) kept ({n_scaled} rescaled), "
                      f"{len(new)} rebound ({n_re:,} rows)", flush=True)
            _first[0] = False
            return [rec["term"] for rec in bound.values()]

        terms = build_terms()
        if verbose and si == 0 and (mq > 0).any():
            print(f"  [wrap] {int((mq > 0).sum()):,} rows with modulus m > 0 "
                  f"(m in [{float(mq[mq > 0].min()):g}, {float(mq[mq > 0].max()):g}]), "
                  f"tau={tau:g}"
                  + (f" -> {tau_end:g} (linear anneal over {n_steps} steps)"
                     if tau_end is not None else " constant")
                  + " (sigma = tau*m); no unwrap precedes the solve", flush=True)

        if bpx:
            from .bpx import BpxPreconditioner

            def level_diag(g, tt=None):
                """GN diagonal of the SAME row system rebuilt on grid g at
                coef = 0 (a quadratic residual's Jacobian there is its linear
                part alone). One BPX level; `tt` reuses a built term set."""
                ax_l = Axes(lo, hi, g, periodic=periodic)
                bs_l = ax_l.bases()
                nl = int(np.prod(g)) * n_channels
                own = tt is None
                if own:
                    if per_stage:
                        rr_, oo_ = rows(g)
                        xl, ol, wl, sl, cl, fl, ml, _ = unpack(rr_, oo_, g)
                    else:
                        xl, ol, wl, sl, cl, fl, ml, oo_ = x, op, w, s, c, fz, mq, ops
                    tt = [t_ for t_, _ in make_terms(ax_l, bs_l, g, xl, ol, wl, sl,
                                                     cl, fl, ml, oo_, False)]
                    for t_ in tt:
                        if hasattr(t_, "set_tau"):
                            t_.set_tau(_tau_now[0])
                zb = ctx.buffer(nl * 4); zb.zero()
                db = ctx.buffer(nl * 4); db.zero()
                for t_ in tt:
                    t_.accumulate_diag(zb, db)
                dmax = float(_MaxReduce(ctx, nl)(db)) + 1e-30
                if own:
                    tt = None                 # release the level's terms
                return dict(ext=tuple(b.primal_extent for b in bs_l),
                            diag=db, dmax=dmax, grid=g)

            lv = [level_diag(g) for g in lv_grids[:-1]]
            lv.append(level_diag(grid, tt=terms))
            opt.set_preconditioner(
                BpxPreconditioner(ctx, extents, n_channels, lv,
                                  floor_rel=bpx_floor))
            if verbose:
                print(f"  [bpx] {len(lv)} levels "
                      f"{[tuple(L['grid']) for L in lv]}, floor_rel={bpx_floor:g}"
                      f"  ({time.time()-t0:.0f}s)", flush=True)

        _anyfuzz = bool((fz > 0).any())
        _seed0 = 1 + seed + si * 10007
        loss_buf = ctx.buffer(4)

        def loss_fn():
            loss_buf.zero()
            for t in terms:
                t.loss(opt.coef, loss_buf)
            return float(loss_buf.download(np.float32, 1)[0])

        # THE FLOOR, measured — but not at the cold start: at coef = 0 every
        # physics residual vanishes for every jitter draw, so the probe must run
        # at a non-trivial point. It runs after the first accepted step: two
        # gradients at the same coefficients and seed (arithmetic), and with
        # jittered rows two more seeds (statistical).
        floor = None

        def _measure_floor():
            def _grad(seed_):
                for _t in terms:
                    if _anyfuzz and hasattr(_t, "set_seed"):
                        _t.set_seed(seed_)
                gb = ctx.buffer(4 * n); gb.zero()
                for _t in terms:
                    _t.accumulate(opt.coef, gb)
                return gb.download(np.float32, n).astype(np.float64)
            sd = _seed0 + 1
            ga, gb_ = _grad(sd), _grad(sd)
            gnorm = max(np.linalg.norm(ga), 1e-30)
            fl = float(np.linalg.norm(ga - gb_) / gnorm)
            if _anyfuzz:
                gc = _grad(sd + 100003)
                fl = max(fl, float(np.linalg.norm(ga - gc) / gnorm))
                for _t in terms:                       # restore the step's draw
                    if hasattr(_t, "set_seed"):
                        _t.set_seed(_seed0 + 1)
            if verbose:
                print(f"  [stop] gradient floor {fl:.2e}"
                      f"{' (arithmetic + jitter)' if _anyfuzz else ' (arithmetic)'}; "
                      f"stop when mean(pred/L) over {stop_window} steps <= {stop_rel:.1e}"
                      + ("  WARNING: stop_rel <= floor, thresholding noise" if stop_rel <= fl else ""),
                      flush=True)
            return fl
        pred_rel = []; _rej = 0; steps_used = n_steps; eff = []

        for _k in range(n_steps):
            if resample_every and _k and per_stage and _k % resample_every == 0:
                stage_rows, ops = rows(grid)
                x, op, w, s, c, fz, mq, row_order = unpack(stage_rows, ops, grid)
                terms = None            # release before allocating the next set
                terms = build_terms()
                opt.last_loss = None    # the objective moved; re-evaluate it
            if tau_end is not None and n_steps > 1:
                _tk = float(tau) + (float(tau_end) - float(tau)) * _k / (n_steps - 1)
                if _tk != _tau_now[0]:
                    _tau_now[0] = _tk
                    for _t in terms:
                        if hasattr(_t, "set_tau"):
                            _t.set_tau(_tk)
                    opt.last_loss = None
            if _anyfuzz:
                for _t in terms:
                    if hasattr(_t, "set_seed"):
                        _t.set_seed(_seed0 + _k)
                opt.last_loss = None
            _, accepted, _ = opt.step(terms, loss_fn, cg_iters=cg_iters)
            if minibatch:
                eff.append(terms[0].eff_batch[0])
            if callback is not None:
                callback(grid, _k + 1, opt)
            if accepted and opt.last_pred is not None:
                pred_rel.append(opt.last_pred / max(abs(opt.last_loss), 1e-30)); _rej = 0
            else:
                pred_rel.append(np.nan); _rej += 1
            if stop_rel > 0.0 and floor is None and accepted:
                floor = _measure_floor()
                opt.last_loss = None if _anyfuzz else opt.last_loss
            if stop_rel > 0.0:
                recent = [v for v in pred_rel[-stop_window:] if np.isfinite(v)]
                if len(recent) == stop_window and float(np.mean(recent)) <= stop_rel:
                    steps_used = _k + 1
                    if verbose:
                        print(f"  [stop] stabilised at step {steps_used}: mean(pred/L) "
                              f"{float(np.mean(recent)):.2e} <= {stop_rel:.1e}", flush=True)
                    break
                if _rej >= 2 and opt.mu_rel >= 1e3:
                    steps_used = _k + 1
                    if verbose:
                        print(f"  [stop] no descent direction at step {steps_used} "
                              f"(two rejections at maximum damping)", flush=True)
                    break
        coef, prev = opt.get_coef(), grid
        losses.append(opt.last_loss)
        eff_hist.append(list(eff))
        if verbose and eff and max(eff) > 1:
            print(f"  [batch] effective batch {np.mean(eff):.1f}/{max(1, len(x) // int(minibatch))} "
                  f"buckets mean (min {min(eff)}, max {max(eff)})", flush=True)
        if verbose:
            print(f"  [stage {grid}] loss {opt.last_loss:.4g}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    diag = {"stage_losses": losses, "seconds": time.time() - t0,
            "row_order": row_order, "pred_rel": pred_rel, "steps_used": steps_used,
            "floor": floor, "eff_batch": eff_hist}
    return FitResult(ctx, coef, axes, bases, extents, losses,
                     n_channels=n_channels, diagnostics=diag)
