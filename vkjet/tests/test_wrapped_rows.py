"""Wrapped-Gaussian rows: the contract (x, op, payload, m) with r ≡ 0 (mod m).

  numpy   soft_unwrap is the derivative of wrapped_half_sq; both m-periodic;
          m = 0 is the identity; tau -> 0 is the hard sawtooth
  gpu     generic referee AND per-op JIT reproduce the oracle on rows whose
          residual spans several wraps (loss, and FD of the loss vs the gradient)
  m = 0   the wrapped kernels reproduce the plain LS numbers exactly
  parity  per-op JIT vs generic on mixed-m rows (verify_generated)
"""
import unittest

import numpy as np

from vkjet.data import data_operator
from vkjet.eqrow import (EqRowTerm, NCH, NCPR, soft_unwrap, wrapped_half_sq,
                         gather_fields_jet, row_residuals_oracle)
from vkjet import genkernel as gk
from vkjet.tests import context, has_compiler, needs_compiler
from vkjet.tests._fixtures import setup


def _rows(bases, rng, n=96):
    x = gk.verify_points(bases, rng, n)
    ops, oid = data_operator()
    c = np.zeros((n, NCPR), np.float32)
    c[:, 2] = 1.0                                 # r = w - s (unit z covector)
    w = rng.uniform(0.5, 1.5, n).astype(np.float32)
    s = rng.uniform(-1.5, 1.5, n).astype(np.float32)   # spans several wraps
    return x, ops, np.zeros(n, np.int32), w, s, c


def _gpu_loss_grad(ctx, term, cb, nco):
    lb = ctx.buffer(4); gb = ctx.buffer(nco * 4)
    lb.zero(); term.loss(cb, lb)
    gb.zero(); term.accumulate(cb, gb)
    return float(lb.download(np.float32, 1)[0]), gb.download(np.float32, nco)


class TestWrappedOracle(unittest.TestCase):
    def test_numpy_oracle(self):
        rng = np.random.default_rng(0)
        r = rng.uniform(-3, 3, 4000)
        for tau in (0.0, 0.1, 0.3, 0.5, -1.0):
            m, h = 1.0, 1e-6
            fd = (wrapped_half_sq(r + h, m, tau) - wrapped_half_sq(r - h, m, tau)) / (2 * h)
            g = soft_unwrap(r, m, tau)
            self.assertLess(np.abs(fd - g).max(), 2e-5, tau)
            self.assertTrue(np.allclose(soft_unwrap(r + 2 * m, m, tau), g, atol=1e-9))
            self.assertTrue(np.allclose(wrapped_half_sq(r - 3 * m, m, tau),
                                        wrapped_half_sq(r, m, tau), atol=1e-9))
        k = 2 * np.pi                                       # tau < 0: pure cosine
        self.assertTrue(np.allclose(soft_unwrap(r, 1.0, -1.0), np.sin(k * r) / k))
        self.assertTrue(np.array_equal(soft_unwrap(r, 0.0, 0.3), r))     # m = 0 identity
        self.assertTrue(np.array_equal(wrapped_half_sq(r, 0.0, 0.3), 0.5 * r * r))
        m = 0.7                                             # tau -> 0: hard sawtooth
        wr = r - m * np.round(r / m)
        ok = np.abs(np.abs(wr) - m / 2) > 0.05
        self.assertTrue(np.allclose(soft_unwrap(r, m, 0.0)[ok], wr[ok], atol=1e-6))
        self.assertTrue(np.allclose(wrapped_half_sq(r, m, 0.0)[ok], 0.5 * wr[ok] ** 2, atol=1e-6))
        # r~ vanishes at the half-wrap and is odd
        self.assertLess(abs(soft_unwrap(np.array([m / 2]), m, 0.3)[0]), 1e-4 * m)
        self.assertTrue(np.allclose(soft_unwrap(-r, m, 0.3), -soft_unwrap(r, m, 0.3)))


class TestWrappedGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = context()
        cls.axes, cls.bases, cls.iw, cls.nco = setup()

    def tiers(self):
        yield "generic", lambda: EqRowTerm(self.ctx, self.bases, self.iw, self.ops)
        if has_compiler():
            yield "jit", lambda: gk.GeneratedRowTerm(self.ctx, self.bases, self.iw, self.ops, 0)

    def residuals(self, x, op, w, s, c, C):
        fields = gather_fields_jet(self.bases, self.iw, x, C.astype(np.float64)
                                   .reshape([b.primal_extent for b in self.bases] + [NCH]))
        return row_residuals_oracle(fields, self.ops, op, w, s, c)

    def test_gpu_vs_oracle(self):
        rng = np.random.default_rng(3)
        x, self.ops, op, w, s, c = _rows(self.bases, rng)
        n = len(x)
        C = (rng.standard_normal(self.nco) * 0.6).astype(np.float32)
        mod = np.where(rng.random(n) < 0.7, 0.5, 0.0).astype(np.float32)   # m = 2·venc
        r = self.residuals(x, op, w, s, c, C)
        self.assertTrue((np.abs(r[mod > 0]) > mod[mod > 0]).any(), "rows must span > 1 wrap")
        cb = self.ctx.buffer(self.nco * 4); cb.upload(C)
        for tau in (0.25, -1.0):
            L_ref = float(np.sum(w * wrapped_half_sq(r, mod, tau)))
            for name, mk in self.tiers():
                with self.subTest(tier=name, tau=tau):
                    t = mk(); t.tau = tau
                    t.bind_batch(x, op, w, s, c, modulus=mod)
                    L, g = _gpu_loss_grad(self.ctx, t, cb, self.nco)
                    self.assertLess(abs(L - L_ref), 2e-4 * abs(L_ref), (L, L_ref))
                    v = rng.standard_normal(self.nco).astype(np.float32); v /= np.linalg.norm(v)
                    h = 2e-3
                    cp = self.ctx.buffer(self.nco * 4); cp.upload(C + h * v)
                    cm = self.ctx.buffer(self.nco * 4); cm.upload(C - h * v)
                    fd = (_gpu_loss_grad(self.ctx, t, cp, self.nco)[0]
                          - _gpu_loss_grad(self.ctx, t, cm, self.nco)[0]) / (2 * h)
                    gv = float(g.astype(np.float64) @ v)
                    self.assertLess(abs(fd - gv), 2e-2 * max(abs(fd), 1e-3), (fd, gv))

    def test_m0_identity(self):
        """With m = 0 everywhere the wrapped kernels equal the plain LS kernels
        for any tau: same loss, same gradient (to atomic order)."""
        rng = np.random.default_rng(5)
        x, self.ops, op, w, s, c = _rows(self.bases, rng)
        C = (rng.standard_normal(self.nco) * 0.6).astype(np.float32)
        cb = self.ctx.buffer(self.nco * 4); cb.upload(C)
        r = self.residuals(x, op, w, s, c, C)
        L_ref = float(0.5 * np.sum(w * r * r))
        for name, mk in self.tiers():
            with self.subTest(tier=name):
                a = mk(); a.bind_batch(x, op, w, s, c)                       # no modulus
                b = mk(); b.tau = 0.4
                b.bind_batch(x, op, w, s, c, modulus=np.zeros(len(x), np.float32))
                La, ga = _gpu_loss_grad(self.ctx, a, cb, self.nco)
                Lb, gb = _gpu_loss_grad(self.ctx, b, cb, self.nco)
                self.assertLess(abs(La - L_ref), 1e-4 * abs(L_ref))
                self.assertLess(abs(La - Lb), 1e-6 * abs(La))
                self.assertLess(np.linalg.norm(ga - gb), 1e-5 * np.linalg.norm(ga))

    @needs_compiler
    def test_parity_mixed_m(self):
        ops, _ = data_operator()
        t = gk.GeneratedRowTerm(self.ctx, self.bases, self.iw, ops, 0)
        gk.verify_generated(self.ctx, t, self.bases, self.iw, ops, 0, force=True)


if __name__ == "__main__":
    unittest.main()
