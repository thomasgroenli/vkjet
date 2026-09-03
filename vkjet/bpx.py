"""GPU-resident BPX multilevel preconditioner — retires the coarse→fine ladder.

    C⁻¹r = Σ_l P_l · (P_lᵀ r) / (diag_l + floor_l)

applied every CG iteration, so ONE cold solve at the finest grid replaces a
stage schedule. Measured on the CFD phantom: beats the ladder at equal
wall-clock AND at equal loss (+0.04 u / +0.06 v at matched objective), so it is
not merely a less-converged ladder — the multilevel prior picks a better point
among equally data-consistent solutions.

The transfers are the tensor product of the Greville-aligned per-axis
interpolation factors, applied ONE AXIS AT A TIME. The assembled 4-D form costs
2^nd nonzeros per FINE node per level (~2.3 GB per level at a 48³×80 grid, plus
a 141M-element host transpose), which is what kept BPX off production grids;
separable needs kilobytes of tables plus two shared scratch buffers.

Per-level diagonals are taken at coef = 0 (a quadratic residual's Jacobian there
is its linear part alone). Re-deriving them at the current coefficients was
measured NEUTRAL (±0.001) for +21–35% wall-clock and is deliberately absent:
what changes with the coefficients is the advective Hessian, whose contribution
is off-diagonal spatial coupling — which a diagonal cannot represent anyway.
"""
import os
import struct

import numpy as np

from .context import Context, STORAGE
from .rowfit import _axis_resize_matrix

SHADER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "shaders", "spv")
AXIS_SPV = os.path.join(SHADER_DIR, "axis_csr.spv")
VEC_PDIV_SPV = os.path.join(SHADER_DIR, "vec_pdiv.spv")

# Each level is damped RELATIVE TO ITSELF, which is what un-freezes the gap
# modes a global-μ nodal diagonal over-damps. Swept at the production grid:
# flat plateau over 1e-2..3e-2, degrading below 3e-3 (less damping drives the
# LOSS down but the correlation and jet amplitude with it).
FLOOR_REL = 1e-2


def _axis_csr(ext_in, ext_out, order=4):
    """(ptr, col, val) of one axis factor M (ext_out × ext_in) and of Mᵀ."""
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
    """Prolongation P (coarse→fine) and its exact adjoint Pᵀ, applied as a
    sequence of per-axis factors. Pᵀ must be the transpose — not an independent
    fine→coarse resize — so that C⁻¹ stays symmetric and CG remains valid."""

    def __init__(self, ctx, ext_c, ext_f, nch, prog, scratch, order=4):
        self.ctx = ctx; self.prog = prog; self.nch = int(nch)
        self.ec = tuple(int(v) for v in ext_c)
        self.ef = tuple(int(v) for v in ext_f)
        self.scratch = scratch
        self.axes = []
        for d in range(len(self.ec)):
            (pf, cf, vf), (pt, ct, vt) = _axis_csr(self.ec[d], self.ef[d], order)
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


class BpxPreconditioner:
    """Multilevel C⁻¹ for GaussNewtonCG (plug in via opt.set_preconditioner).

    levels: [{ext: tuple, diag: device buffer (prod(ext)·nch), dmax: float}],
    any order; the level whose ext == ext_fine is the identity (no transfer)."""

    def __init__(self, ctx: Context, ext_fine, n_channels, levels, order=4):
        self.ctx = ctx; self.nch = int(n_channels); self.levels = levels
        self.ext_f = tuple(int(v) for v in ext_fine)
        n_f = int(np.prod(self.ext_f))
        self.pdiv_prog = ctx.program(VEC_PDIV_SPV, bindings=[STORAGE] * 4)
        self.pmeta = ctx.buffer(12, device_local=False)
        self.axis_prog = self.scratch = None
        if any(tuple(L["ext"]) != self.ext_f for L in levels):
            self.axis_prog = ctx.program(AXIS_SPV, bindings=[STORAGE] * 6)
            self.scratch = [ctx.buffer(n_f * self.nch * 4) for _ in range(2)]
        for L in levels:
            L["ext"] = tuple(int(v) for v in L["ext"])
            L["nl"] = int(np.prod(L["ext"])) * self.nch
            L["floor"] = FLOOR_REL * L["dmax"]
            if L["ext"] == self.ext_f:
                L["T"] = None
            else:
                L["T"] = SeparableTransfer(ctx, L["ext"], self.ext_f, self.nch,
                                           self.axis_prog, self.scratch, order)
                L["rbuf"] = ctx.buffer(L["nl"] * 4)

    def _pdiv(self, x, d, a, z, n):                  # z = x/(d + a) over n elems
        self.pmeta.upload(struct.pack("<i2f", n, float(a), 0.0))
        self.ctx.run(self.pdiv_prog, [x, d, z, self.pmeta],
                     groups=min((n + 255) // 256, 4096))

    def apply(self, r_buf, z_buf):
        """z = Σ_l P_l (P_lᵀ r)/(diag_l + floor_l). Writes z_buf."""
        wrote = False
        for L in self.levels:                        # identity (finest) first
            if L["T"] is None:
                self._pdiv(r_buf, L["diag"], L["floor"], z_buf, L["nl"])
                wrote = True
        for L in self.levels:                        # then coarse corrections
            if L["T"] is not None:
                L["T"].restrict(r_buf, L["rbuf"])
                self._pdiv(L["rbuf"], L["diag"], L["floor"], L["rbuf"], L["nl"])
                L["T"].prolong(L["rbuf"], z_buf, accum=wrote)
                wrote = True
