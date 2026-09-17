"""End-to-end: fit_rows recovers a known field, and JIT == generic.

  1. the JIT tier and the pure-generic tier produce the same objective
     (execution is an implementation detail, never a semantic one)
  2. a divergence-free field fitted from directional measurements + a
     continuity constraint is recovered
  3. the forward evaluator round-trips the fitted coefficients
"""
import unittest

import numpy as np

from vkjet import (OperatorTable, make_rows, merge_row_sets, data_operator, fit_rows,
                   SLOT_DX, SLOT_DY, SLOT_DZ)
from vkjet.tests import context
from vkjet.tests._fixtures import divfree_truth, directional_rows

LO, HI = (0., 0., 0., 0.), (1., 1., 1., 1.)
GRID, STAGES = (4, 4, 4, 4), 2


def build(seed=0, n_data=40000, n_col=8000):
    rng = np.random.default_rng(seed)
    xd = rng.uniform(0, 1, (n_data, 4)).astype(np.float32)
    dops, did = data_operator()
    drows = directional_rows(rng, xd, divfree_truth(xd), did)
    pops = OperatorTable()
    cid = pops.add_op("continuity", lin=[(SLOT_DX, 0, 1.0), (SLOT_DY, 1, 1.0), (SLOT_DZ, 2, 1.0)])
    xc = rng.uniform(0, 1, (n_col, 4)).astype(np.float32)
    prows = make_rows(xc, np.full(n_col, cid, np.int32), np.full(n_col, 0.1, np.float32),
                      np.zeros(n_col, np.float32))
    return merge_row_sets((drows, dops), (prows, pops))


class TestFitRows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = context()
        rows, ops = build()
        cls.out = {mode: fit_rows(rows, ops, lo=LO, hi=HI, base_grid=GRID, n_stages=STAGES,
                                  steps=12, cg_iters=8, dispatch=mode, ctx=cls.ctx, verbose=False)
                   for mode in (True, False)}

    def test_jit_equals_generic(self):
        a, b = self.out[True].stage_losses[-1], self.out[False].stage_losses[-1]
        self.assertLess(abs(a - b) / max(abs(b), 1e-12), 1e-3, (a, b))

    def test_recovery_through_forward(self):
        rng = np.random.default_rng(99)
        xq = rng.uniform(0.05, 0.95, (10000, 4)).astype(np.float32)
        Ut = divfree_truth(xq)
        for mode in (True, False):
            with self.subTest(dispatch=mode):
                Uh = self.out[mode].forward(xq)
                self.assertEqual(Uh.shape, (len(xq), 5))
                cc = [float(np.corrcoef(Uh[:, k], Ut[:, k])[0, 1]) for k in range(3)]
                rmse = float(np.sqrt(((Uh[:, :3] - Ut[:, :3]) ** 2).mean()))
                nrmse = rmse / float(np.sqrt((Ut[:, :3] ** 2).mean()))
                self.assertGreater(min(cc), 0.97, cc)
                self.assertLess(nrmse, 0.15, nrmse)


if __name__ == "__main__":
    unittest.main()
