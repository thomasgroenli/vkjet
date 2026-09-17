"""The generic jet-row kernels against the float64 numpy oracle, on a random
operator table exercising every slot class (value, first, pure second, mixed),
payload-indexed coefficients, pure-linear, pure-quadratic and mixed operators,
with random per-row weights, targets and payloads.

  loss   GPU vs oracle
  grad   GPU directional derivatives vs FD of the oracle loss
  hvp    GPU bilinear form uᵀHv vs Σ w·(Ju)(Jv) with J by residual FD; symmetry
  diag   GPU diag vs eᵢᵀHeᵢ from the GPU hvp
  scale  set_scale linearity
  file   save_rows / load_rows / merge_row_sets round trip with op-id remap
"""
import os
import unittest

import numpy as np

from vkjet.basis import Basis1D
from vkjet.data import save_rows, load_rows, merge_row_sets, make_rows
from vkjet.eqrow import (OperatorTable, EqRowTerm, gather_fields_jet,
                         row_residuals_oracle, row_loss_oracle,
                         SLOT_VAL, SLOT_DT, SLOT_DX, SLOT_DY, SLOT_DZ,
                         SLOT_DTT, SLOT_DXX, SLOT_DYZ, SLOT_DTX)
from vkjet.tests import context, scratch_dir

GRID = (5, 6, 6, 7)
IW = [1.3, 0.9, 1.1, 0.7]
N = 60
SCALE = 0.7


def build_ops():
    ops = OperatorTable()
    ops.add_op("data", lin=[(SLOT_VAL, ch, 1.0, ch) for ch in range(5)])
    ops.add_op("lin-pde", lin=[(SLOT_DX, 0, 1.0), (SLOT_DY, 1, 1.0),
                               (SLOT_DZ, 2, 1.0), (SLOT_DTT, 3, 0.3),
                               (SLOT_DXX, 0, -0.02), (SLOT_DYZ, 4, 0.11),
                               (SLOT_DTX, 2, -0.4)])
    ops.add_op("quad", quad=[(SLOT_VAL, 0, SLOT_DX, 2, 1.0),
                             (SLOT_VAL, 1, SLOT_DY, 2, 1.0),
                             (SLOT_VAL, 2, SLOT_DZ, 2, 1.0),
                             (SLOT_VAL, 4, SLOT_DTT, 4, 0.5, 3)])
    ops.add_op("mixed", lin=[(SLOT_DT, 4, 0.8), (SLOT_VAL, 3, 1.0, 6)],
               quad=[(SLOT_DX, 4, SLOT_VAL, 0, 1.0),
                     (SLOT_DY, 4, SLOT_VAL, 1, 1.0, 1)])
    return ops


class TestEqRow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = context()
        rng = cls.rng = np.random.default_rng(0)
        cls.bases = [Basis1D.bspline(list(range(g + 1)), 4) for g in GRID]
        ext = tuple(b.primal_extent for b in cls.bases)
        cls.shape = ext + (5,)
        cls.nco = int(np.prod(ext)) * 5
        cls.C = rng.standard_normal(cls.shape).astype(np.float32) * 0.3
        cls.x = rng.uniform(0, 1, (N, 4)).astype(np.float32) * np.array(GRID, np.float32)
        cls.ops = build_ops()
        cls.op = rng.integers(0, cls.ops.n_ops, N).astype(np.int32)
        cls.w = rng.uniform(0.2, 2.0, N).astype(np.float32)
        cls.s = rng.standard_normal(N).astype(np.float32) * 0.2
        cls.c = rng.standard_normal((N, 8)).astype(np.float32)
        cls.term = EqRowTerm(cls.ctx, cls.bases, IW, cls.ops)
        cls.term.bind_batch(cls.x, cls.op, cls.w, cls.s, cls.c, scale=SCALE)
        cls.cbuf = cls.ctx.buffer(cls.nco * 4); cls.cbuf.upload(cls.C.reshape(-1))
        cls.lbuf = cls.ctx.buffer(4)
        cls.vbuf = cls.ctx.buffer(cls.nco * 4); cls.obuf = cls.ctx.buffer(cls.nco * 4)

    def oracle_loss(self, Cv):
        return row_loss_oracle(self.bases, IW, self.x, Cv, self.ops, self.op,
                               self.w, self.s, self.c, SCALE)

    def gpu_loss(self, Cv):
        self.cbuf.upload(np.ascontiguousarray(Cv, np.float32).reshape(-1))
        self.lbuf.zero(); self.term.loss(self.cbuf, self.lbuf)
        return float(self.lbuf.download(np.float32, 1)[0])

    def gpu_hvp(self, v):
        self.vbuf.upload(np.asarray(v, np.float32)); self.obuf.zero()
        self.term.hvp(self.cbuf, self.vbuf, self.obuf)
        return self.obuf.download(np.float32, self.nco).astype(np.float64)

    def test_loss(self):
        Lg, Lo = self.gpu_loss(self.C), self.oracle_loss(self.C.astype(np.float64))
        self.assertLess(abs(Lg - Lo) / abs(Lo), 1e-4, (Lg, Lo))

    def test_grad_vs_oracle_fd(self):
        gbuf = self.ctx.buffer(self.nco * 4); gbuf.zero()
        self.cbuf.upload(self.C.reshape(-1))
        self.term.accumulate(self.cbuf, gbuf)
        g = gbuf.download(np.float32, self.nco).astype(np.float64)
        C64 = self.C.astype(np.float64); eps = 1e-4
        for _ in range(4):
            dv = self.rng.standard_normal(self.nco); dv /= np.linalg.norm(dv)
            fd = (self.oracle_loss(C64 + eps * dv.reshape(self.shape))
                  - self.oracle_loss(C64 - eps * dv.reshape(self.shape))) / (2 * eps)
            self.assertLess(abs(fd - g @ dv) / max(abs(fd), 1e-12), 5e-3)

    def test_hvp_bilinear_and_symmetric(self):
        C64 = self.C.astype(np.float64)

        def resid(Cv):
            return row_residuals_oracle(gather_fields_jet(self.bases, IW, self.x, Cv),
                                        self.ops, self.op, self.w, self.s, self.c)

        def J(u, eps=1e-5):
            return (resid(C64 + eps * u.reshape(self.shape))
                    - resid(C64 - eps * u.reshape(self.shape))) / (2 * eps)

        self.cbuf.upload(self.C.reshape(-1))
        for _ in range(3):
            u = self.rng.standard_normal(self.nco) * 0.1
            v = self.rng.standard_normal(self.nco) * 0.1
            Hv, Hu = self.gpu_hvp(v), self.gpu_hvp(u)
            ref = float(SCALE * (self.w.astype(np.float64) * J(u) * J(v)).sum())
            self.assertLess(abs(u @ Hv - ref) / max(abs(ref), 1e-12), 5e-3)
            self.assertLess(abs(v @ Hu - u @ Hv) / max(abs(u @ Hv), 1e-12), 1e-3)

    def test_diag_vs_hvp(self):
        self.cbuf.upload(self.C.reshape(-1))
        dbuf = self.ctx.buffer(self.nco * 4); dbuf.zero()
        self.term.accumulate_diag(self.cbuf, dbuf)
        diag = dbuf.download(np.float32, self.nco).astype(np.float64)
        for i in self.rng.choice(self.nco, 8, replace=False):
            e = np.zeros(self.nco, np.float32); e[i] = 1.0
            hii = self.gpu_hvp(e)[i]
            self.assertLess(abs(hii - diag[i]) / max(abs(hii), abs(diag[i]), 1e-9), 1e-3)

    def test_set_scale_linearity(self):
        L1 = self.gpu_loss(self.C)
        self.term.set_scale(2 * SCALE)
        try:
            L2 = self.gpu_loss(self.C)
        finally:
            self.term.set_scale(SCALE)
        self.assertLess(abs(L2 - 2 * L1) / abs(2 * L1), 1e-5)

    def test_file_round_trip_and_merge_remap(self):
        rows = make_rows(self.x, self.op, self.w, self.s, self.c)
        Lo = self.oracle_loss(self.C.astype(np.float64))
        p = os.path.join(scratch_dir(), "a.npz")
        save_rows(p, rows, self.ops)
        rows_l, ops_l = load_rows(p)
        self.assertTrue(np.allclose(rows_l["x"], rows["x"]))
        self.assertTrue((rows_l["op"] == rows["op"]).all())
        Lr = row_loss_oracle(self.bases, IW, rows_l["x"], self.C.astype(np.float64),
                             ops_l, rows_l["op"], rows_l["w"], rows_l["s"], rows_l["c"], SCALE)
        self.assertLess(abs(Lr - Lo) / abs(Lo), 1e-6)
        # a second set: the SAME quad op in a different position + one new op
        ops2 = OperatorTable()
        ops2.add_op("quad", self.ops.lin[2], self.ops.quad[2])
        ops2.add_op("extra", lin=[(SLOT_DX, 3, 2.0)])
        rows2 = make_rows(self.x[:5], np.array([0, 1, 0, 1, 0], np.int32),
                          self.w[:5], self.s[:5], self.c[:5])
        rows_m, ops_m = merge_row_sets((rows, self.ops), (rows2, ops2))
        self.assertEqual(ops_m.n_ops, self.ops.n_ops + 1)      # "quad" deduped
        self.assertEqual(len(rows_m), len(rows) + 5)
        f2 = gather_fields_jet(self.bases, IW, self.x[:5], self.C.astype(np.float64))
        r_direct = row_residuals_oracle(f2, ops2, rows2["op"], rows2["w"], rows2["s"], rows2["c"])
        tail = rows_m[len(rows):]
        r_merged = row_residuals_oracle(f2, ops_m, tail["op"], tail["w"], tail["s"], tail["c"])
        self.assertTrue(np.allclose(r_direct, r_merged))


if __name__ == "__main__":
    unittest.main()
