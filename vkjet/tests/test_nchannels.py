"""The channel count is DECLARED, not compiled in: a specialisation constant
carried by the OperatorTable (and so by the row file), because the table
references channel indices. A table at a channel count other than 5 gives
exactly the numpy oracle (NCH = 5 is covered by test_eqrow)."""
import unittest

import numpy as np

from vkjet.data import Axes, make_rows
from vkjet.eqrow import (EqRowTerm, OperatorTable, SLOT_VAL, SLOT_DT, SLOT_DX, SLOT_DXX,
                         gather_fields_jet, row_residuals_oracle)
from vkjet.tests import context

LO, HI, GRID = [0.0, 0.0, 0.0, 0.0], [1.0, 0.06, 0.06, 0.10], (6, 6, 6, 8)


class TestNChannels(unittest.TestCase):
    def check(self, nch, rng):
        ctx = context()
        axes = Axes(LO, HI, GRID, periodic=(True, False, False, False))
        bases = axes.bases()
        ext = np.asarray(HI) - np.asarray(LO)
        iw = [GRID[k] / ext[k] for k in range(4)]
        n = int(np.prod(GRID)) * nch
        ops = OperatorTable(n_channels=nch)
        top = nch - 1                       # exercise the HIGHEST channel index
        ops.add_op("mixed", lin=[(SLOT_VAL, 0, 1.0), (SLOT_DT, top, 0.7), (SLOT_DXX, min(2, top), -0.3)],
                   quad=[(SLOT_VAL, 0, SLOT_DX, top, 0.5)])
        ops.add_op("payload-covector", lin=[(SLOT_VAL, ch, 1.0, ch) for ch in range(min(nch, 8))],
                   const=((-1.0, 7),) if nch <= 7 else ())
        N = 60                              # the identity is per row; the f64
                                            # oracle is a per-row Python gather
        x = np.stack([rng.uniform(l, h, N) for l, h in zip(LO, HI)], 1).astype(np.float32)
        op = rng.integers(0, ops.n_ops, N).astype(np.int32)
        w = np.abs(rng.normal(1.0, 0.2, N)).astype(np.float32)
        s = rng.normal(size=N).astype(np.float32)
        c = rng.normal(size=(N, 8)).astype(np.float32)
        make_rows(x, op, w, s, c)
        t = EqRowTerm(ctx, bases, iw, ops)
        self.assertEqual(t.nch, nch)
        t.bind_batch(axes.encode(x), op, w, s, c)
        coef = (rng.normal(size=n) * 0.1).astype(np.float32)
        cb = ctx.buffer(n * 4); cb.upload(coef)
        lb = ctx.buffer(4); lb.zero(); t.loss(cb, lb)
        g = ctx.buffer(n * 4); g.zero(); t.accumulate(cb, g)
        L_gpu = float(lb.download(np.float32, 1)[0])
        g_gpu = g.download(np.float32, n)
        extents = tuple(b.primal_extent for b in bases)

        def oracle(cc):
            f = gather_fields_jet(bases, iw, axes.encode(x), cc.reshape(extents + (nch,)))
            r = row_residuals_oracle(f, ops, op, w, s, c)
            return float(0.5 * np.sum(w.astype(np.float64) * r * r))

        L_or = oracle(coef.astype(np.float64))
        self.assertLess(abs(L_gpu - L_or) / max(abs(L_or), 1e-30), 1e-5)
        for i in rng.choice(n, 1, replace=False):     # gradient by central differences
            eps = 1e-3; dl = []
            for sgn in (1.0, -1.0):
                cc = coef.astype(np.float64).copy(); cc[i] += sgn * eps
                dl.append(oracle(cc))
            fd = (dl[0] - dl[1]) / (2 * eps)
            self.assertLess(abs(fd - g_gpu[i]) / (abs(fd) + 1.0), 1e-3, (nch, i))
        del t

    def test_nch_3_and_7(self):
        rng = np.random.default_rng(0)
        for nch in (3, 7):
            with self.subTest(nch=nch):
                self.check(nch, rng)


if __name__ == "__main__":
    unittest.main()
