"""Deflation: an explicit coarse space LEARNED from test vectors.

    C⁻¹ = M⁻¹ + W (WᵀHW)⁻¹ Wᵀ

structurally one more additive level on top of any base preconditioner M — its
prolongation is W (n×m, m small) and its coarse operator is a dense m×m matrix
solved on the host. It is the one algebraic-multigrid idea that survives this
discretization: AMG proper needs the assembled matrix (cubic tensor-product
B-splines give 7⁴·5 = 12,005 nonzeros per row ⇒ ~4.2 TB at the production grid),
whereas ADAPTIVE / bootstrap coarse-space selection needs only matvecs.

W is found the bootstrap way: relax on the HOMOGENEOUS problem with the base
preconditioner, v ← v − M⁻¹(H v), and keep what refuses to converge. Those are
by construction the modes the current preconditioner leaves slow — which, on a
problem whose unobserved subspace is decided by the solver's implicit prior, is
exactly the subspace worth handling explicitly.

Physics-agnostic by construction: it consumes an `hvp(v, out)` callable and
knows nothing about what the rows mean, so it applies to any jet-row system, not
only flow.
"""
import struct

import numpy as np

from .context import STORAGE
from .optim import VEC_DOT_SPV, VEC_AXPBY_SPV


class Deflation:
    """Wraps a base preconditioner (anything with .apply(r, z)) with a learned
    coarse space. Cost per CG iteration: m dot products + m axpys over n."""

    def __init__(self, ctx, base, n, hvp, m=8, n_relax=8, seed=0, ridge=1e-6,
                 verbose=True):
        self.ctx = ctx; self.base = base; self.n = int(n); self.m = int(m)
        self.dot_p = ctx.program(VEC_DOT_SPV, bindings=[STORAGE] * 4)
        self.axpby_p = ctx.program(VEC_AXPBY_SPV, bindings=[STORAGE] * 4)
        self.vmeta = ctx.buffer(12, device_local=False)
        self.scalar = ctx.buffer(4)
        self.t1 = ctx.buffer(self.n * 4)
        self.t2 = ctx.buffer(self.n * 4)
        self.W = [ctx.buffer(self.n * 4) for _ in range(self.m)]
        self._build(hvp, int(n_relax), int(seed), float(ridge), verbose)

    # -- vector primitives ------------------------------------------------- #
    def _g(self):
        return min((self.n + 255) // 256, 4096)

    def _vm(self, a, b):
        self.vmeta.upload(struct.pack("<i2f", self.n, float(a), float(b)))
        return self.vmeta

    def _dot(self, a, b):
        self.scalar.zero()
        self.ctx.run(self.dot_p, [a, b, self.scalar, self._vm(0.0, 0.0)],
                     groups=self._g())
        return float(self.scalar.download(np.float32, 1)[0])

    def _axpby(self, x, y, a, b, z):          # z = a·x + b·y
        self.ctx.run(self.axpby_p, [x, y, z, self._vm(a, b)], groups=self._g())

    # -- construction ------------------------------------------------------ #
    def _build(self, hvp, n_relax, seed, ridge, verbose):
        rng = np.random.default_rng(seed)
        for j in range(self.m):
            v = rng.normal(size=self.n).astype(np.float32)
            v /= np.linalg.norm(v)
            self.W[j].upload(v)
        # DAMPING FACTOR. The relaxation below is Richardson on M⁻¹H, which is
        # only contractive for generalized eigenvalues below 2 — undamped, modes
        # ABOVE that amplify and the coarse space fills with the largest modes
        # instead of the slowest (measured: 20x enrichment instead of 1000x).
        # omega = 1/lambda_max makes every factor (1 - mu/lambda_max) lie in
        # [0, 1), so the SMALLEST mu decays slowest and survives — which is the
        # subspace we are after. lambda_max by power iteration on M⁻¹H.
        v = rng.normal(size=self.n).astype(np.float32)
        self.t1.upload(v / np.linalg.norm(v))
        lam = 1.0
        for _ in range(12):
            self.t2.zero()
            hvp(self.t1, self.t2)
            self.base.apply(self.t2, self.t1)         # t1 = M⁻¹H·v
            nrm = np.sqrt(max(self._dot(self.t1, self.t1), 1e-300))
            lam = nrm
            self._axpby(self.t1, self.t1, 1.0 / nrm, 0.0, self.t1)
        self.lam_max = float(lam)
        om = 1.0 / max(self.lam_max, 1e-30)
        # bootstrap relaxation on H v = 0: what survives is what M leaves slow
        for _ in range(n_relax):
            for j in range(self.m):
                self.t1.zero()
                hvp(self.W[j], self.t1)               # t1 = H·w
                self.base.apply(self.t1, self.t2)     # t2 = M⁻¹H·w
                self._axpby(self.W[j], self.t2, 1.0, -om, self.W[j])
            self._orthonormalize()
        self._orthonormalize()
        # coarse operator E = WᵀHW (dense, m×m) — m hvps
        E = np.zeros((self.m, self.m))
        for j in range(self.m):
            self.t1.zero()
            hvp(self.W[j], self.t1)
            for i in range(self.m):
                E[i, j] = self._dot(self.W[i], self.t1)
        E = 0.5 * (E + E.T)
        tr = max(float(np.trace(E)) / self.m, 1e-300)
        E += ridge * tr * np.eye(self.m)               # keep E invertible
        self.E = E
        self.Einv = np.linalg.inv(E)
        w = np.linalg.eigvalsh(E)
        if verbose:
            print(f"    [deflate] m={self.m} test vectors, {n_relax} relaxations "
                  f"damped by 1/lambda_max={om:.3e}; "
                  f"WᵀHW eigenvalues {w.min():.3e} .. {w.max():.3e} "
                  f"(cond {w.max()/max(w.min(), 1e-300):.1e})", flush=True)

    def _orthonormalize(self):
        """Modified Gram-Schmidt on the device (m² dots, m² axpys — m is small)."""
        for j in range(self.m):
            for i in range(j):
                p = self._dot(self.W[i], self.W[j])
                self._axpby(self.W[i], self.W[j], -p, 1.0, self.W[j])
            nrm = np.sqrt(max(self._dot(self.W[j], self.W[j]), 1e-300))
            self._axpby(self.W[j], self.W[j], 1.0 / nrm, 0.0, self.W[j])

    def refresh(self, diag_fine):
        """Forward to the base so deflation composes with level-diag refresh.
        W itself is NOT relearned here — rebuilding the coarse space mid-CG would
        change the preconditioner between iterations and break Krylov."""
        if hasattr(self.base, "refresh"):
            self.base.refresh(diag_fine)

    # -- application ------------------------------------------------------- #
    def apply(self, r_buf, z_buf):
        """z = M⁻¹r + W E⁻¹ Wᵀ r."""
        self.base.apply(r_buf, z_buf)
        c = np.array([self._dot(self.W[j], r_buf) for j in range(self.m)])
        a = self.Einv @ c
        for j in range(self.m):
            self._axpby(self.W[j], z_buf, float(a[j]), 1.0, z_buf)
