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
from .optim import GaussNewtonCG


# --------------------------------------------------------------------------- #
# coarse -> fine warm start                                                   #
# --------------------------------------------------------------------------- #
def _axis_resize_matrix(n_in: int, n_out: int, order: int = 4) -> np.ndarray:
    """(n_out x n_in) PERIODIC, Greville-aligned linear-interpolation matrix.

    A B-spline control point k influences the field centered at its Greville
    abscissa k-(order-1)/2, and that shift must be matched across scales or the
    warm-started field is translated by ~(order-1)/2 intervals (worse than a
    cold start). Identity when n_in == n_out.
    """
    if n_in == 1:
        return np.ones((n_out, 1), dtype=np.float64)
    g = (order - 1) / 2.0
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
def _jit_terms(ctx, axes, bases, grid, x, op, w, s, c, ops, iw, verbose):
    """Compile a specialised kernel per operator where possible.

    Payload-free operators that share an IDENTICAL point set fuse into one
    shared-gather kernel; the rest get a per-operator kernel. Returns
    (terms, handled_mask); anything not handled is left to the generic term.
    """
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

    # --- grouped: payload-free ops on a shared point set -------------------- #
    payload_free = [
        int(k) for k in np.unique(op)
        if all(e[3] < 0 for e in ops.lin[k]) and all(q[5] < 0 for q in ops.quad[k])]
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
        try:
            gt = GroupedRowTerm(ctx, bases, iw, ops, gids, compiler=comp)
            verify_grouped(ctx, gt, bases, iw, ops, gids)
            xp = x[members[0][1]]
            W = np.stack([w[r] for _, r in members], 1)
            S = np.stack([s[r] for _, r in members], 1)
            gt.bind_points(axes.encode(xp), W, S)
            terms.append(gt)
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
            gt.bind_batch(axes.encode(x[m]), np.zeros(int(m.sum()), np.int32),
                          w[m], s[m], c[m])
            terms.append(gt)
            handled |= m
        except Exception as ex:
            if verbose:
                print(f"    [dispatch] jit '{ops.names[int(k)]}' failed "
                      f"({type(ex).__name__}: {ex}) — generic", flush=True)
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
             bpx=False, ctx=None, verbose=True):
    """Fit a spline field from jet rows (array pair, or a save_rows path).

    dispatch=True  JIT-generated kernels, generic for anything they cannot take
    dispatch=False pure generic EqRowTerm (the semantic reference)

    Returns a :class:`FitResult`.
    """
    t0 = time.time()
    if isinstance(rows, (str, os.PathLike)):
        rows, ops = load_rows(rows)
    if ops is not None:
        # the ROW FILE declares how many channels its operators reference;
        # the n_channels argument is only the fallback for a table that
        # predates the field
        n_channels = int(getattr(ops, "n_channels", n_channels))
    assert ops is not None and lo is not None and hi is not None
    own_ctx = ctx is None
    ctx = ctx or Context()

    x = np.ascontiguousarray(rows["x"], np.float32)
    op = np.ascontiguousarray(rows["op"], np.int32)
    w = np.ascontiguousarray(rows["w"], np.float32)
    s = np.ascontiguousarray(rows["s"], np.float32)
    c = np.ascontiguousarray(rows["c"], np.float32)

    grids = [tuple(g * (1 << k) if g > 1 else 1 for g in base_grid)
             for k in range(n_stages)]
    if steps is None:
        steps = (25, 25, 30, 25, 25)[:n_stages]
    if np.isscalar(steps):
        steps = (int(steps),) * n_stages
    ext = np.asarray(hi, np.float64) - np.asarray(lo, np.float64)

    def cell_ids(pts, grid):
        ax = Axes(lo, hi, grid, periodic=periodic)
        e = np.floor(ax.encode(pts)).astype(np.int64)
        cid = np.zeros(len(pts), np.int64)
        for k, g in enumerate(grid):
            cid = cid * g + np.clip(e[:, k], 0, g - 1)
        return cid

    # one finest-grid stable sort by (cell, op): scatter locality at every
    # coarser stage (dyadic) + warp-uniform operator ids inside cells
    key = cell_ids(x, grids[-1]) * (ops.n_ops + 1) + op
    row_order = np.argsort(key, kind="stable")
    x, op, w, s, c = (a[row_order] for a in (x, op, w, s, c))
    x = np.ascontiguousarray(x)

    # BPX: the ladder GRIDS become preconditioner LEVELS and the solve is ONE
    # cold run at the finest of them. The stage schedule and its steps-per-stage
    # split disappear; a scalar `steps` is then the TOTAL budget. Measured on the
    # CFD phantom this beats the ladder at equal wall-clock AND at equal loss.
    lv_grids = ()
    if bpx:
        lv_grids = tuple(grids)
        grids = [grids[-1]]
        steps = (int(sum(steps)),)

    coef, prev = None, None
    losses = []
    for si, (grid, n_steps) in enumerate(zip(grids, steps)):
        axes = Axes(lo, hi, grid, periodic=periodic)
        bases = axes.bases()
        extents = tuple(b.primal_extent for b in bases)
        n = int(np.prod(grid)) * n_channels
        iw = [grid[k] / ext[k] for k in range(len(grid))]   # pure chain rule
        coef = (np.zeros(n, np.float32) if coef is None
                else multilinear_resize(coef, prev, grid, n_channels))
        opt = GaussNewtonCG(ctx, n); opt.set_coef(coef)

        if dispatch:
            terms, handled = _jit_terms(ctx, axes, bases, grid, x, op, w, s, c,
                                        ops, iw, verbose and si == 0)
            if not handled.all():
                gen = EqRowTerm(ctx, bases, iw, ops)
                r = ~handled
                gen.bind_batch(axes.encode(x[r]), op[r], w[r], s[r], c[r])
                terms.append(gen)
            if verbose and si == 0:
                print(f"    [dispatch] jit {int(handled.sum()):,}  "
                      f"generic {int((~handled).sum()):,}", flush=True)
        else:
            term = EqRowTerm(ctx, bases, iw, ops)
            term.bind_batch(axes.encode(x), op, w, s, c)
            terms = [term]

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
                    iw_l = [g[k] / ext[k] for k in range(len(g))]
                    tt = [EqRowTerm(ctx, bs_l, iw_l, ops)]
                    tt[0].bind_batch(ax_l.encode(x), op, w, s, c)
                zb = ctx.buffer(nl * 4); zb.zero()
                db = ctx.buffer(nl * 4); db.zero()
                for t_ in tt:
                    t_.accumulate_diag(zb, db)
                dmax = float(db.download(np.float32, nl).max()) + 1e-30
                if own:
                    tt = None                 # release the level's terms
                return dict(ext=tuple(b.primal_extent for b in bs_l),
                            diag=db, dmax=dmax, grid=g)

            lv = [level_diag(g) for g in lv_grids[:-1]]
            lv.append(level_diag(grid, tt=terms))
            opt.set_preconditioner(
                BpxPreconditioner(ctx, extents, n_channels, lv))
            if verbose:
                print(f"  [bpx] {len(lv)} levels "
                      f"{[tuple(L['grid']) for L in lv]}  "
                      f"({time.time()-t0:.0f}s)", flush=True)

        loss_buf = ctx.buffer(4)

        def loss_fn():
            loss_buf.zero()
            for t in terms:
                t.loss(opt.coef, loss_buf)
            return float(loss_buf.download(np.float32, 1)[0])

        for _ in range(n_steps):
            opt.step(terms, loss_fn, cg_iters=cg_iters)
        coef, prev = opt.get_coef(), grid
        losses.append(opt.last_loss)
        if verbose:
            print(f"  [stage {grid}] loss {opt.last_loss:.4g}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    diag = {"stage_losses": losses, "seconds": time.time() - t0,
            "row_order": row_order}
    return FitResult(ctx, coef, axes, bases, extents, losses,
                     n_channels=n_channels, diagnostics=diag)
