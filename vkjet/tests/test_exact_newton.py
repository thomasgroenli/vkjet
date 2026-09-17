"""Exact-polynomial optimiser extensions: the loss of plain (m = 0, unjittered)
rows is an exact QUARTIC in the coefficients — data residuals linear, a
quadratic operator's residual quadratic — so L(c + αv) is a quartic in α and
the gradient is a cubic, whose central difference is the exact Hessian action
at any ε.

  A  quartic exactness: 5 loss evaluations pin L(c+αv), predicted at held-out α
  B  ε-independence of the central-difference Newton HVP (only true for a cubic)
  C  curvature ground truth: uᵀHu == 2·a₂ of the quartic along u; off-diagonal
     by polarisation
  D  symmetry uᵀHv == vᵀHu
  F  integration: line-search steps are monotone; ls+newton is competitive
"""
import unittest

import numpy as np

from vkjet.eqrow import EqRowTerm, OperatorTable
from vkjet.optim import GaussNewtonCG
from vkjet.tests import context
from vkjet.tests._fixtures import setup, sample_operators, random_rows

GRID = (4, 4, 4, 4)
FIT_A = (0.0, 0.25, 0.5, 1.0, 2.0)
TEST_A = (0.75, 1.5, 3.0)


def quartic_fit(alphas, losses):
    """Exact quartic through (0, L0) and the (α, L) samples → a1..a4."""
    L0 = losses[0.0]
    aa = [a for a in alphas if a != 0.0]
    A = np.array([[a ** k for k in range(1, 5)] for a in aa], np.float64)
    return np.linalg.solve(A, np.array([losses[a] - L0 for a in aa], np.float64))


class TestExactNewton(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = context()
        rng = cls.rng = np.random.default_rng(7)
        cls.axes, cls.bases, cls.iw, cls.n = setup(GRID)
        ext = [b.primal_extent for b in cls.bases]
        ops, ids = sample_operators()
        phys = OperatorTable(); data = OperatorTable()
        for k in (ids["mom0"], ids["mom1"], ids["cont"]):
            phys.add_op(ops.names[k], ops.lin[k], ops.quad[k])
        data.add_op("data", ops.lin[ids["data"]])
        cls.xp = random_rows(rng, phys, 300, extents=ext)
        cls.xd = random_rows(rng, data, 500, extents=ext)
        cls.pt = EqRowTerm(cls.ctx, cls.bases, cls.iw, phys); cls.pt.bind_batch(*cls.xp)
        cls.dt = EqRowTerm(cls.ctx, cls.bases, cls.iw, data); cls.dt.bind_batch(*cls.xd)
        cls.terms = [cls.dt, cls.pt]
        cls.loss_buf = cls.ctx.buffer(4); cls.cbuf = cls.ctx.buffer(cls.n * 4)
        cls.c = (0.5 * rng.standard_normal(cls.n)).astype(np.float32)
        cls.v = (0.3 * rng.standard_normal(cls.n)).astype(np.float32)
        cls.u = (0.3 * rng.standard_normal(cls.n)).astype(np.float32)
        cls.opt = GaussNewtonCG(cls.ctx, cls.n); cls.opt.set_coef(cls.c)
        cls.opt._cnorm = float(np.linalg.norm(cls.c.astype(np.float64)))
        cls.pb = cls.ctx.buffer(cls.n * 4); cls.ob = cls.ctx.buffer(cls.n * 4)

    def loss_at(self, c):
        self.cbuf.upload(np.asarray(c, np.float32)); self.loss_buf.zero()
        for t in self.terms:
            t.loss(self.cbuf, self.loss_buf)
        return float(self.loss_buf.download(np.float32, 1)[0])

    def newton_hvp(self, p, eps_rel=0.1):
        self.pb.upload(np.asarray(p, np.float32)); self.ob.zero()
        self.opt._hvp_full(self.terms, self.pb, self.ob, eps_rel)
        return self.ob.download(np.float32, self.n).astype(np.float64)

    def a2_along(self, w):
        return quartic_fit(FIT_A, {a: self.loss_at(self.c + a * w) for a in FIT_A})[1]

    def test_a_quartic_exact(self):
        L = {a: self.loss_at(self.c + a * self.v) for a in FIT_A + TEST_A}
        cf = quartic_fit(FIT_A, L)
        spread = max(abs(L[a] - L[0.0]) for a in TEST_A)
        err = max(abs((((cf[3] * a + cf[2]) * a + cf[1]) * a + cf[0]) * a - (L[a] - L[0.0]))
                  for a in TEST_A) / spread
        self.assertLess(err, 1e-3, f"held-out rel err {err:.2e}")

    def test_b_eps_independence(self):
        h1, h2 = self.newton_hvp(self.v, 0.05), self.newton_hvp(self.v, 0.5)
        self.assertLess(np.linalg.norm(h1 - h2) / np.linalg.norm(h1), 1e-2)

    def test_c_curvature_ground_truth(self):
        u, v = self.u.astype(np.float64), self.v.astype(np.float64)
        uHu, ref = float(u @ self.newton_hvp(self.u)), 2.0 * self.a2_along(self.u)
        self.assertLess(abs(uHu - ref) / abs(ref), 1e-2, (uHu, ref))
        uHv = float(u @ self.newton_hvp(self.v))
        ref = 0.5 * (self.a2_along(self.u + self.v) - self.a2_along(self.u - self.v))
        self.assertLess(abs(uHv - ref) / (abs(ref) + 1e-30), 5e-2, (uHv, ref))

    def test_d_symmetry(self):
        a = float(self.u.astype(np.float64) @ self.newton_hvp(self.v))
        b = float(self.v.astype(np.float64) @ self.newton_hvp(self.u))
        self.assertLess(abs(a - b) / abs(a), 1e-2, (a, b))

    def test_f_integration(self):
        def fit(**kw):
            o = GaussNewtonCG(self.ctx, self.n); o.set_coef(np.zeros(self.n, np.float32))

            def lf():
                self.loss_buf.zero()
                for t in self.terms:
                    t.loss(o.coef, self.loss_buf)
                return float(self.loss_buf.download(np.float32, 1)[0])
            return [o.step(self.terms, lf, cg_iters=8, **kw)[0] for _ in range(8)]

        h_leg = fit()
        h_ls = fit(line_search=True)
        h_lsn = fit(line_search=True, newton_eps=0.1)
        for h in (h_ls, h_lsn):
            self.assertTrue(all(b <= a * (1 + 1e-6) for a, b in zip(h, h[1:])), h)
        self.assertLessEqual(h_lsn[-1], h_leg[-1] * 2.0, (h_leg[-1], h_ls[-1], h_lsn[-1]))


if __name__ == "__main__":
    unittest.main()
