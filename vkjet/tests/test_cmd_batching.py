"""Command-buffer batching: the CG solve recorded once into a CommandSequence
and re-submitted equals the per-dispatch solve.

  A  one solve: batched δ == sequential δ (atomic-order noise only)
  B  a new μ re-uploads one meta buffer and re-submits the SAME sequence
  C  rebinding a term's batch changes the key: re-recorded, still correct
  D  full fits, batched (default) vs forced-sequential, track to fp32 noise
"""
import unittest

import numpy as np

from vkjet.eqrow import EqRowTerm, OperatorTable
from vkjet.optim import GaussNewtonCG
from vkjet.tests import context, rel
from vkjet.tests._fixtures import setup, sample_operators, random_rows

GRID = (4, 4, 4, 4)


class TestCmdBatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = context()
        cls.rng = np.random.default_rng(9)
        cls.axes, cls.bases, cls.iw, cls.n = setup(GRID)
        ext = [b.primal_extent for b in cls.bases]
        ops, ids = sample_operators()
        # a physics term (quadratic) and a data term (linear), as separate terms
        cls.phys = OperatorTable(); cls.data = OperatorTable()
        for k in (ids["mom0"], ids["cont"]):
            cls.phys.add_op(ops.names[k], ops.lin[k], ops.quad[k])
        cls.data.add_op("data", ops.lin[ids["data"]])
        xp = random_rows(cls.rng, cls.phys, 400, extents=ext)
        cls.xd = random_rows(cls.rng, cls.data, 600, extents=ext)
        cls.pt = EqRowTerm(cls.ctx, cls.bases, cls.iw, cls.phys); cls.pt.bind_batch(*xp)
        cls.dt = EqRowTerm(cls.ctx, cls.bases, cls.iw, cls.data); cls.dt.bind_batch(*cls.xd)
        cls.terms = [cls.dt, cls.pt]
        cls.c0 = (0.4 * cls.rng.standard_normal(cls.n)).astype(np.float32)
        cls.loss_buf = cls.ctx.buffer(4)

    def prep(self, opt):
        opt.set_coef(self.c0)
        opt.grad.zero(); opt.diag.zero()
        for t in self.terms:
            t.accumulate(opt.coef, opt.grad)
            t.accumulate_diag(opt.coef, opt.diag)
        opt.diag_max = opt._maxred(opt.diag)

    def solve_both(self, opt, mu):
        opt._pcg(self.terms, mu, 6, 1e-3)
        d_seq = opt.delta.download(np.float32, self.n).copy()
        opt._pcg_batched(self.terms, mu, 6)
        d_bat = opt.delta.download(np.float32, self.n).copy()
        return d_seq, d_bat

    def test_solve_reuse_and_rebind(self):
        opt = GaussNewtonCG(self.ctx, self.n); self.prep(opt)
        mu = 1e-2 * opt.diag_max
        d_seq, d_bat = self.solve_both(opt, mu)                 # A
        self.assertLess(rel(d_bat, d_seq), 1e-5)
        key = opt._seq_key
        d_seq, d_bat = self.solve_both(opt, 0.37 * opt.diag_max)   # B: new μ, same sequence
        self.assertLess(rel(d_bat, d_seq), 1e-5)
        self.assertEqual(opt._seq_key, key)
        x, op, w, s, c = self.xd                                # C: rebind → re-record
        self.dt.bind_batch(x, op, w, self.rng.standard_normal(len(s)).astype(np.float32), c)
        self.prep(opt)
        d_seq, d_bat = self.solve_both(opt, mu)
        self.assertLess(rel(d_bat, d_seq), 1e-5)
        self.assertNotEqual(opt._seq_key, key)

    def test_fit_equivalence(self):
        def fit(force_seq):
            o = GaussNewtonCG(self.ctx, self.n); o.set_coef(np.zeros(self.n, np.float32))
            if force_seq:
                o._pcg_batched = lambda t, m, c: o._pcg(t, m, c, 1e-3)

            def lf():
                self.loss_buf.zero()
                for t in self.terms:
                    t.loss(o.coef, self.loss_buf)
                return float(self.loss_buf.download(np.float32, 1)[0])
            return np.array([o.step(self.terms, lf, cg_iters=6)[0] for _ in range(8)])

        h_seq, h_bat = fit(True), fit(False)
        dev = float(np.abs(h_bat - h_seq).max() / h_seq[-1])
        self.assertLess(dev, 1e-3, f"loss trajectories deviate {dev:.2e}")


if __name__ == "__main__":
    unittest.main()
