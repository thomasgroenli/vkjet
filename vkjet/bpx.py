"""GPU-resident BPX (multilevel) preconditioner for GaussNewtonCG.

Discards the explicit coarse→fine stage ladder: one cold solve, the multiscale conditioning
supplied by the preconditioner  C⁻¹ r = Σ_l P_l · (P_lᵀ r)/(diag_l + floor_l)  applied each CG
iteration. P_l (prolong level-l→finest) and P_lᵀ (restrict) are the tensor product of the
Greville-aligned linear interpolation matrices (trainer._axis_resize_matrix), assembled once
as sparse CSR on the SPATIAL grid (periodic wrap + alignment baked into the indices) and
applied by the csr_matvec kernel — fully device-resident, no host round-trip per CG iter.

See docs/2026-06-08-…md §5b: BPX conditions cross-scale; pairing it with GN-CG's truncated CG
(within-level) reaches staged-multiscale quality from a cold start with no stage schedule.
"""
import math
import os
import struct

import numpy as np

from .context import Context, STORAGE
from .rowfit import _axis_resize_matrix

SHADER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "shaders", "spv")
CSR_SPV = os.path.join(SHADER_DIR, "csr_matvec.spv")
AXIS_SPV = os.path.join(SHADER_DIR, "axis_csr.spv")
VEC_PDIV_SPV = os.path.join(SHADER_DIR, "vec_pdiv.spv")


def _axis_nz(ext_in, ext_out, order):
    """Per-axis interpolation as (idx[ext_out,2], w[ext_out,2]) — ≤2 nonzeros/row."""
    M = _axis_resize_matrix(ext_in, ext_out, order)            # (ext_out, ext_in)
    idx = np.zeros((ext_out, 2), np.int64); w = np.zeros((ext_out, 2))
    for o in range(ext_out):
        nz = np.nonzero(M[o])[0]
        if len(nz) == 1:
            idx[o] = nz[0]; w[o] = [M[o, nz[0]], 0.0]
        else:
            idx[o] = nz[:2]; w[o] = M[o, nz[:2]]
    return idx, w


def build_resize_csr(ext_c, ext_f, order=4):
    """Spatial-grid CSR of the prolongation P (fine←coarse) and its transpose Pᵀ
    (coarse←fine), as the tensor product of the per-axis interpolation factors.
    Returns (P, Pt), each = (ptr int32, col int32, val float32, n_rows)."""
    nd = len(ext_c)
    ax = [_axis_nz(ext_c[d], ext_f[d], order) for d in range(nd)]
    n_f = int(np.prod(ext_f)); n_c = int(np.prod(ext_c))
    cstr = [int(np.prod(ext_c[d + 1:])) for d in range(nd)]      # coarse C-order strides
    fco = np.unravel_index(np.arange(n_f), ext_f)                # fine coord per axis
    ncomb = 1 << nd
    cols = np.zeros((n_f, ncomb), np.int64); vals = np.ones((n_f, ncomb))
    for c in range(ncomb):
        cf = np.zeros(n_f, np.int64); wv = np.ones(n_f)
        for d in range(nd):
            sel = (c >> d) & 1
            cf += ax[d][0][fco[d], sel] * cstr[d]
            wv *= ax[d][1][fco[d], sel]
        cols[:, c] = cf; vals[:, c] = wv
    ptr = np.arange(0, n_f * ncomb + 1, ncomb, dtype=np.int32)
    col = cols.reshape(-1).astype(np.int32); val = vals.reshape(-1).astype(np.float32)
    P = (ptr, col, val, n_f)
    fr = np.repeat(np.arange(n_f, dtype=np.int32), ncomb)        # transpose → coarse rows
    ordr = np.argsort(col, kind="stable")
    ptr_t = np.searchsorted(col[ordr], np.arange(n_c + 1)).astype(np.int32)
    Pt = (ptr_t, fr[ordr].copy(), val[ordr].copy(), n_c)
    return P, Pt


def _axis_csr(ext_in, ext_out, order=4):
    """(ptr, col, val) of one axis factor M (ext_out x ext_in) and of Mᵀ."""
    M = _axis_resize_matrix(ext_in, ext_out, order)

    def csr(A):
        nz = [np.nonzero(A[r])[0] for r in range(A.shape[0])]
        ptr = np.zeros(A.shape[0] + 1, np.int32)
        ptr[1:] = np.cumsum([len(z) for z in nz])
        col = (np.concatenate(nz).astype(np.int32) if nz
               else np.zeros(0, np.int32))
        val = np.concatenate([A[r][nz[r]] for r in range(A.shape[0])]
                             or [np.zeros(0)]).astype(np.float32)
        return ptr, col, val
    return csr(M), csr(np.ascontiguousarray(M.T))


class SeparableTransfer:
    """The tensor-product transfer applied ONE AXIS AT A TIME (axis_csr kernel).

    Mathematically identical to the assembled 4-D CSR of build_resize_csr — same
    factors, same order — but the tables are O(sum extent) instead of
    O(2^nd · prod(extent)): at the production (48,48,48,80) grid the assembled
    form needs ~2.3 GB per level and a 141M-element host sort to transpose,
    which is what kept BPX off the finest production ladder. Costs two shared
    ping-pong scratch buffers (prod(ext_f)·nch floats each) for the whole
    preconditioner, not per level."""

    def __init__(self, ctx, ext_c, ext_f, nch, prog, scratch, order=4,
                 squared=False):
        """squared=True squares every transfer weight, giving (P.P) instead of P.
        Restricting the FINE GN diagonal with it yields diag(P'HP) with the
        off-diagonal terms dropped — the Galerkin coarse diagonal, matrix-free
        and at the CURRENT coefficients, instead of rebuilding the whole row
        system on each coarse grid at theta = 0."""
        self.ctx = ctx; self.prog = prog; self.nch = int(nch)
        self.ec = tuple(int(v) for v in ext_c)
        self.ef = tuple(int(v) for v in ext_f)
        self.scratch = scratch
        self.axes = []
        for d in range(len(self.ec)):
            (pf, cf, vf), (pt, ct, vt) = _axis_csr(self.ec[d], self.ef[d], order)
            if squared:
                vf = (vf.astype(np.float64) ** 2).astype(np.float32)
                vt = (vt.astype(np.float64) ** 2).astype(np.float32)
            self.axes.append({"fwd": self._up(pf, cf, vf),
                              "bwd": self._up(pt, ct, vt)})
        self.active = [d for d in range(len(self.ec)) if self.ec[d] != self.ef[d]]
        self.meta = [ctx.buffer(20, device_local=False)
                     for _ in range(2 * len(self.ec))]

    def _up(self, ptr, col, val):
        bufs = []
        for a in (ptr, col, val):
            b = self.ctx.buffer(max(a.nbytes, 4)); b.upload(a)
            bufs.append(b)
        return bufs

    def _pass(self, mi, tabs, n_out, n_in, outer, inner, src, dst, accum):
        self.meta[mi].upload(struct.pack("<5i", n_out, n_in, outer, inner,
                                         int(accum)))
        n = outer * n_out * inner
        self.ctx.run(self.prog, [self.meta[mi], tabs[0], tabs[1], tabs[2],
                                 src, dst],
                     groups=min((n + 255) // 256, 4096))

    def _run(self, src, dst, shape_in, shape_out, key, accum):
        if not self.active:
            self.ctx.copy_buffer(src, dst,
                                 int(np.prod(shape_in)) * self.nch * 4)
            return                       # identity level (no transfer)
        sh = list(shape_in)
        cur = src
        for n, d in enumerate(self.active):
            last = n == len(self.active) - 1
            out = dst if last else self.scratch[n % 2]
            outer = int(np.prod(sh[:d])) if d else 1
            inner = int(np.prod(sh[d + 1:])) * self.nch
            self._pass(2 * d + (0 if key == "fwd" else 1),
                       self.axes[d][key], shape_out[d], sh[d], outer, inner,
                       cur, out, accum and last)
            sh[d] = shape_out[d]
            cur = out

    def prolong(self, src, dst, accum=False):    # coarse -> fine
        self._run(src, dst, self.ec, self.ef, "fwd", accum)

    def restrict(self, src, dst):                # fine -> coarse (exact Pᵀ)
        self._run(src, dst, self.ef, self.ec, "bwd", False)


class _Csr:
    """A CSR matrix resident on the GPU + its matvec dispatch."""
    def __init__(self, ctx, csr, nch, prog):
        ptr, col, val, n_rows = csr
        self.ctx = ctx; self.n_rows = int(n_rows); self.nch = int(nch); self.prog = prog
        self.ptr = ctx.buffer(ptr.nbytes); self.ptr.upload(ptr)
        self.col = ctx.buffer(col.nbytes); self.col.upload(col)
        self.val = ctx.buffer(val.nbytes); self.val.upload(val)
        self.meta = ctx.buffer(12, device_local=False)

    def matvec(self, x_buf, y_buf, accum=False):
        self.meta.upload(struct.pack("<3i", self.n_rows, self.nch, int(accum)))
        self.ctx.run(self.prog, [self.meta, self.ptr, self.col, self.val, x_buf, y_buf],
                     groups=min((self.n_rows * self.nch + 255) // 256, 4096))


def make_field_restrictor(ctx, ext_from, ext_to, nch, order=4, scratch=None,
                         prog=None):
    """A field-preserving resize ext_from -> ext_to (NOT the adjoint Pᵀ): the
    coarse COEFFICIENTS whose field approximates the fine one, which is what a
    coarse-level operator must be evaluated at. Its .prolong(src, dst) does the
    transfer."""
    prog = prog or ctx.program(AXIS_SPV, bindings=[STORAGE] * 6)
    if scratch is None:
        n = int(np.prod(ext_from)) * int(nch)
        scratch = [ctx.buffer(n * 4) for _ in range(2)]
    return SeparableTransfer(ctx, ext_from, ext_to, nch, prog, scratch, order)


class _CsrPair:
    """Assembled-CSR transfer behind the SeparableTransfer call shape."""

    def __init__(self, P, Pt):
        self.P = P; self.Pt = Pt

    def prolong(self, src, dst, accum=False):
        self.P.matvec(src, dst, accum=accum)

    def restrict(self, src, dst):
        self.Pt.matvec(src, dst)


class BpxPreconditioner:
    """Multilevel preconditioner C⁻¹ for GaussNewtonCG (set via opt.set_preconditioner).

    levels: list of {ext: tuple, diag: device buffer (prod(ext)*nch), dmax: float}, any order;
    a level whose ext == ext_fine is the identity (no transfer). floor_rel sets the per-level
    floor (diag_l + floor_rel·max diag_l) — each level damped relative to ITSELF, which is what
    un-freezes the gap modes a global-μ nodal diagonal over-damps.
    """
    def __init__(self, ctx: Context, ext_fine, n_channels, levels, floor_rel=1e-2, order=4,
                 transfer="auto", level_weight=0.0, diag_mode="rebuild",
                 refresh_every=0):
        self.ctx = ctx; self.nch = int(n_channels); self.levels = levels
        self.ext_f = tuple(int(v) for v in ext_fine)
        n_f = int(np.prod(self.ext_f))
        # "auto": the assembled 4-D CSR costs 2^nd nonzeros per FINE node per level
        # (~2.3 GB/level at (48,48,48,80), plus a 141M-element host transpose sort),
        # so it only stays affordable on small grids. Everything at or below the
        # historically validated sizes keeps the assembled path bit-for-bit.
        if transfer == "auto":
            transfer = "csr" if n_f <= 2_000_000 else "separable"
        if diag_mode == "restrict":
            transfer = "separable"      # the squared restrictor is a separable object
        self.transfer = transfer
        self.csr_prog = ctx.program(CSR_SPV, bindings=[STORAGE] * 6)
        self.pdiv_prog = ctx.program(VEC_PDIV_SPV, bindings=[STORAGE] * 4)
        self.pmeta = ctx.buffer(12, device_local=False)
        self.axis_prog = self.scratch = None
        self._maxr = {}
        if transfer == "separable" and any(tuple(L["ext"]) != self.ext_f for L in levels):
            self.axis_prog = ctx.program(AXIS_SPV, bindings=[STORAGE] * 6)
            self.scratch = [ctx.buffer(n_f * self.nch * 4) for _ in range(2)]
        # Per-level weight omega_j = 2^(alpha·d_j), d_j = dyadic distance from the
        # FINEST grid. alpha = 0 reproduces the flat sum exactly (no scaling applied
        # at all); alpha > 0 emphasizes coarse levels — a stronger smoothness prior on
        # the step — and alpha < 0 emphasizes fine. This is the H^s knob: floor_rel
        # damps WITHIN a level, alpha sets the balance BETWEEN levels, and the two are
        # orthogonal. omega multiplies the whole level term, which is identical to
        # dividing (diag_l + floor_l) by it, so it is folded in once at setup and
        # costs nothing per CG iteration. Dividing the floor too keeps each level
        # damped relative to ITSELF.
        self.level_weight = float(level_weight)
        # diag_mode: "rebuild" = each level's diag is the row system rebuilt on that
        # grid at coef 0 (the validated path); "restrict" = derive every coarse diag
        # from the FINE diag by squared-weight restriction, i.e. the Galerkin coarse
        # diagonal minus off-diagonals. "restrict" costs a few passes over the fine
        # diag instead of rebuilding 18M rows per level, and — unlike "rebuild" — can
        # track the CURRENT coefficients, which is what refresh_every > 0 does.
        self.diag_mode = diag_mode
        self.refresh_every = int(refresh_every)
        self.refresh_hook = None      # caller-supplied exact re-derivation
        self._nrefresh = 0
        self.floor_rel = float(floor_rel)
        if diag_mode == "restrict":
            assert abs(self.level_weight) < 1e-12, (
                "restrict diags and level_weight != 0 both rescale a level; "
                "keep alpha = 0 (the measured optimum) when refreshing")
        for L in levels:
            L["ext"] = tuple(int(v) for v in L["ext"])
            L["nl"] = int(np.prod(L["ext"])) * self.nch
            L["floor"] = floor_rel * L["dmax"]
            d_j = max(math.log2(self.ext_f[k] / L["ext"][k])
                      for k in range(len(self.ext_f))) if L["ext"] != self.ext_f else 0.0
            L["omega"] = 2.0 ** (self.level_weight * d_j)
            if abs(L["omega"] - 1.0) > 1e-12:
                d = L["diag"].download(np.float32, L["nl"]) / L["omega"]
                sb = ctx.buffer(L["nl"] * 4); sb.upload(d.astype(np.float32))
                L["diag"] = sb
                L["floor"] /= L["omega"]
            if L["ext"] == self.ext_f:
                L["T"] = None
            elif transfer == "separable":
                L["T"] = SeparableTransfer(ctx, L["ext"], self.ext_f, self.nch,
                                           self.axis_prog, self.scratch, order)
                L["rbuf"] = ctx.buffer(L["nl"] * 4)
                if diag_mode == "restrict":
                    L["Tsq"] = SeparableTransfer(ctx, L["ext"], self.ext_f,
                                                 self.nch, self.axis_prog,
                                                 self.scratch, order, squared=True)
                    L["diag"] = ctx.buffer(L["nl"] * 4)
            else:
                P, Pt = build_resize_csr(L["ext"], self.ext_f, order)
                Pc = _Csr(ctx, P, self.nch, self.csr_prog)       # coarse→fine
                Ptc = _Csr(ctx, Pt, self.nch, self.csr_prog)     # fine→coarse
                L["T"] = _CsrPair(Pc, Ptc)
                L["rbuf"] = ctx.buffer(L["nl"] * 4)              # restricted scratch

    def set_fine_diag(self, diag_fine):
        """Fill every level's diagonal from the CURRENT fine GN diagonal:
        the finest level aliases it, each coarse level takes the squared-weight
        restriction (the Galerkin diagonal without off-diagonals). Floors are
        re-derived per level so each stays damped relative to ITSELF."""
        for L in self.levels:
            if L["T"] is None:
                L["diag"] = diag_fine
            else:
                L["Tsq"].restrict(diag_fine, L["diag"])
            L["floor"] = self.floor_rel * self._maxred(L["diag"], L["nl"])

    def refresh(self, diag_fine):
        """Called by GaussNewtonCG each step. No-op unless refresh_every > 0.

        Two ways to re-derive the level diagonals at the CURRENT coefficients:
        diag_mode="restrict" takes the squared-weight restriction of the fine
        diag (cheap, but drops off-diagonals — measured to cost more than the
        refresh gains), while refresh_hook lets the caller, which owns the
        per-level terms, recompute them exactly."""
        if self.refresh_every <= 0:
            return
        if self._nrefresh % self.refresh_every == 0:
            if self.refresh_hook is not None:
                self.refresh_hook(self)
            else:
                self.set_fine_diag(diag_fine)
        self._nrefresh += 1

    def relevel(self):
        """Re-derive every level's floor after its diag buffer was rewritten in
        place by a refresh_hook."""
        for L in self.levels:
            L["floor"] = self.floor_rel * self._maxred(L["diag"], L["nl"])

    def _maxred(self, buf, n):
        from .optim import _MaxReduce
        key = int(n)
        if key not in self._maxr:
            self._maxr[key] = _MaxReduce(self.ctx, key)
        return float(self._maxr[key](buf)) + 1e-30

    def _pdiv(self, x, d, a, z, n):                              # z = x/(d + a) over n elems
        self.pmeta.upload(struct.pack("<i2f", n, float(a), 0.0))
        self.ctx.run(self.pdiv_prog, [x, d, z, self.pmeta], groups=min((n + 255) // 256, 4096))

    def apply(self, r_buf, z_buf):
        """z = Σ_l P_l (P_lᵀ r)/(diag_l + floor_l).  Writes z_buf."""
        wrote = False
        for L in self.levels:                                    # identity (finest) first
            if L["T"] is None:
                self._pdiv(r_buf, L["diag"], L["floor"], z_buf, L["nl"]); wrote = True
        for L in self.levels:                                    # then coarse corrections
            if L["T"] is not None:
                L["T"].restrict(r_buf, L["rbuf"])                # restrict r → level l
                self._pdiv(L["rbuf"], L["diag"], L["floor"], L["rbuf"], L["nl"])
                L["T"].prolong(L["rbuf"], z_buf, accum=wrote)    # prolong & sum into z
                wrote = True
