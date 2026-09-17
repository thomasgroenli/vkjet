"""The `s` column is redundant: a residual's ORDER-0 term can live in the
operator table like its order-1 and order-2 terms, with the per-row value
riding the payload.

    r = <L, J(f)> + J'QJ - s          (s column)
    r = <L, J(f)> + J'QJ - c[k]       (SLOT_CONST entry, payload slot k)

These agree on loss, gradient, GN diagonal and HVP — the same arithmetic in a
different place — and both match the numpy oracle. The JIT must SPECIALISE an
order-0 operator, not fall back: in a JIT-only package the generic kernel is
the referee, not a fast path.
"""
import unittest

import numpy as np

from vkjet.data import Axes, rows_from_unified
from vkjet.eqrow import (EqRowTerm, SLOT_CONST, row_residuals_oracle, gather_fields_jet)
from vkjet.tests import context, needs_compiler

LO, HI, GRID = [0.0, 0.0, 0.0, 0.0], [1.0, 0.06, 0.06, 0.10], (6, 6, 6, 8)


def _term(ctx, bases, iw, rows, ops, axes):
    t = EqRowTerm(ctx, bases, iw, ops)
    t.bind_batch(axes.encode(np.ascontiguousarray(rows["x"], np.float32)),
                 np.ascontiguousarray(rows["op"], np.int32),
                 np.ascontiguousarray(rows["w"], np.float32),
                 np.ascontiguousarray(rows["s"], np.float32),
                 np.ascontiguousarray(rows["c"], np.float32))
    return t


def _rows(rng, n):
    x = np.stack([rng.uniform(l, h, n) for l, h in zip(LO, HI)], 1).astype(np.float32)
    d = rng.normal(size=(n, 5)).astype(np.float32)
    s = rng.normal(size=n).astype(np.float32)
    return x, d, s


class TestConstTerm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = context()
        cls.axes = Axes(LO, HI, GRID, periodic=(True, False, False, False))
        cls.bases = cls.axes.bases()
        ext = np.asarray(HI) - np.asarray(LO)
        cls.iw = [GRID[k] / ext[k] for k in range(4)]
        cls.n = int(np.prod(GRID)) * 5

    def test_s_column_equals_order0_term(self):
        rng = np.random.default_rng(0)
        x, d, s = _rows(rng, 3000)
        rows_s, ops_s = rows_from_unified(x, d, s)                    # s column
        rows_c, ops_c = rows_from_unified(x, d, s, target_cix=5)      # order-0 term
        self.assertTrue(any(e[0] == SLOT_CONST for e in ops_c.lin[0]))
        self.assertFalse(any(e[0] == SLOT_CONST for e in ops_s.lin[0]))
        self.assertEqual(float(np.abs(rows_c["s"]).max()), 0.0)
        coef = rng.normal(size=self.n).astype(np.float32) * 0.1
        cbuf = self.ctx.buffer(self.n * 4); cbuf.upload(coef)
        vbuf = self.ctx.buffer(self.n * 4); vbuf.upload(rng.normal(size=self.n).astype(np.float32))
        out = {}
        for tag, (rws, ops) in (("s", (rows_s, ops_s)), ("const", (rows_c, ops_c))):
            t = _term(self.ctx, self.bases, self.iw, rws, ops, self.axes)
            lb = self.ctx.buffer(4); lb.zero(); t.loss(cbuf, lb)
            g = self.ctx.buffer(self.n * 4); g.zero(); t.accumulate(cbuf, g)
            dg = self.ctx.buffer(self.n * 4); dg.zero(); t.accumulate_diag(cbuf, dg)
            hv = self.ctx.buffer(self.n * 4); hv.zero(); t.hvp(cbuf, vbuf, hv)
            out[tag] = (float(lb.download(np.float32, 1)[0]), g.download(np.float32, self.n),
                        dg.download(np.float32, self.n), hv.download(np.float32, self.n))
            del t
        for i, nm in enumerate(("loss", "grad", "diag", "hvp")):
            a, b = out["s"][i], out["const"][i]
            r = (abs(a - b) / max(abs(a), 1e-30) if i == 0
                 else float(np.abs(a - b).max() / (np.abs(a).max() + 1e-30)))
            self.assertLess(r, 1e-6, (nm, r))
        # both match the numpy oracle: the identity is per row, so a subset
        # of the rows through the float64 gather is the whole check
        k = 300
        sub_s, sub_c = rows_s[:k], rows_c[:k]
        flds = gather_fields_jet(self.bases, self.iw, self.axes.encode(x[:k]),
                                 coef.astype(np.float64).reshape(tuple(b.primal_extent for b in self.bases) + (5,)))
        r_s = row_residuals_oracle(flds, ops_s, sub_s["op"], sub_s["w"], sub_s["s"], sub_s["c"])
        r_c = row_residuals_oracle(flds, ops_c, sub_c["op"], sub_c["w"], sub_c["s"], sub_c["c"])
        self.assertLess(float(np.abs(r_s - r_c).max() / (np.abs(r_s).max() + 1e-30)), 1e-9)
        t = _term(self.ctx, self.bases, self.iw, sub_c, ops_c, self.axes)
        lb = self.ctx.buffer(4); lb.zero(); t.loss(cbuf, lb)
        L_gpu = float(lb.download(np.float32, 1)[0])
        L_or = float(0.5 * np.sum(sub_c["w"] * r_c * r_c))
        self.assertLess(abs(L_or - L_gpu) / max(abs(L_or), 1e-30), 1e-5)

    @needs_compiler
    def test_jit_specialises_order0(self):
        from vkjet.genkernel import GeneratedRowTerm, verify_generated, emit_shaders
        rng = np.random.default_rng(4)
        x, d, s = _rows(rng, 1000)
        rows_c, ops_c = rows_from_unified(x, d, s, target_cix=5)
        srcs = emit_shaders(ops_c, 0)[0]
        self.assertNotIn("f15_", srcs["grad"], "emitted a gather for the order-0 sentinel slot")
        gt = GeneratedRowTerm(self.ctx, self.bases, self.iw, ops_c, 0)
        verify_generated(self.ctx, gt, self.bases, self.iw, ops_c, 0)     # parity vs the referee
        del gt


if __name__ == "__main__":
    unittest.main()
