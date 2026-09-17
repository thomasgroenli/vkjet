"""Authoring-side folding: (x, op, w, s, c, m) must define the SAME objective
as the (x, op, payload, m) the solver is handed.

With every coefficient on a payload slot the residual is homogeneous of degree
1 in the payload, so w·r(c)² = r(√w·c)²; a wrapped row's loss is homogeneous of
degree 2 in (r, m), so w·L(r; m) = L(√w·r; √w·m). Checked on a
production-shaped operator set — momentum-like rows with literal coefficients
and quadratic advection, continuity, a per-row-payload (frame) continuity, and
data rows carrying their target in `s` (which must migrate to SLOT_CONST
before it can fold), most of them wrapped with per-row moduli.

  identity  homogenize + fold reproduces the f64 oracle loss, and the GPU
            loss/grad/diag/hvp on the generic referee and the JIT tier, at two
            temperatures (wrapped Gaussian and the pure cosine)
  reweight  reweighting AFTER folding == changing w BEFORE folding
  refusals  s with no target slot; non-homogeneous operator
  gain_of   recovers w where a gain slot exists, NaN where it honestly cannot
  hash      two folds of one system hash alike; the fold is a different file
"""
import unittest

import numpy as np

from vkjet.data import make_rows, data_operator, rows_hash
from vkjet.eqrow import (EqRowTerm, OperatorTable, row_loss_oracle, SLOT_VAL, SLOT_DT,
                         SLOT_DX, SLOT_DY, SLOT_DZ, SLOT_DXX, SLOT_DYY, SLOT_DZZ)
from vkjet import genkernel as gk
from vkjet.rowauthor import homogenize, fold, reweight, gain_of, is_homogeneous
from vkjet.tests import context, has_compiler, rel
from vkjet.tests._fixtures import setup

LO, HI, GRID = (0., 0., 0., 0.), (1., .06, .06, .10), (4, 4, 4, 6)
NC, ND = 24, 60             # the identity is per row and the f64 oracle is a
                            # per-row Python gather: enough to hit every operator


def physics_ops(Lh=0.1, nu=1e-3):
    ops = OperatorTable()
    adv = lambda ch: [(SLOT_VAL, 0, SLOT_DX, ch, Lh), (SLOT_VAL, 1, SLOT_DY, ch, Lh),
                      (SLOT_VAL, 2, SLOT_DZ, ch, Lh)]
    lap = lambda ch: [(SLOT_DXX, ch, -nu * Lh * Lh), (SLOT_DYY, ch, -nu * Lh * Lh),
                      (SLOT_DZZ, ch, -nu * Lh * Lh)]
    grad = [SLOT_DX, SLOT_DY, SLOT_DZ]
    ids = {}
    for ch, nm in enumerate("uvw"):
        ids[f"mom-{nm}"] = ops.add_op(f"mom-{nm}", lin=[(SLOT_DT, ch, Lh), (grad[ch], 3, Lh)] + lap(ch),
                                      quad=adv(ch))
    ids["cont"] = ops.add_op("cont", lin=[(SLOT_DX, 0, Lh), (SLOT_DY, 1, Lh), (SLOT_DZ, 2, Lh)])
    ids["cont-frame"] = ops.add_op("cont-frame",           # five payload coefficients: homogeneous already
                                   lin=[(SLOT_DX, 0, Lh, 0), (SLOT_DZ, 0, Lh, 1), (SLOT_VAL, 0, Lh, 2),
                                        (SLOT_DX, 1, Lh, 3), (SLOT_DZ, 1, Lh, 4)])
    return ops, ids


def build(rng):
    ops, ids = physics_ops()
    ops, did = data_operator(ops)
    sets = []
    for nm, oid in ids.items():
        x = rng.uniform(LO, HI, (NC, 4)).astype(np.float32)
        pay = None
        if nm == "cont-frame":
            a = rng.uniform(-0.5, 0.5, NC); rr = rng.uniform(0.02, 0.1, NC)
            pay = np.stack([np.sin(a), np.cos(a), 1 / rr, np.cos(a), -np.sin(a)], 1)
        sets.append(make_rows(x, np.full(NC, oid, np.int32), rng.uniform(0.2, 5.0, NC).astype(np.float32),
                              np.zeros(NC, np.float32), pay))
    x = rng.uniform(LO, HI, (ND, 4)).astype(np.float32)
    c = np.zeros((ND, 5), np.float32); c[:, :3] = rng.standard_normal((ND, 3))
    mod = np.where(rng.random(ND) < 0.7, 0.5, 0.0).astype(np.float32)   # m = 2 venc
    sets.append(make_rows(x, np.full(ND, did, np.int32), rng.uniform(0.2, 5.0, ND).astype(np.float32),
                          rng.uniform(-1.5, 1.5, ND).astype(np.float32), c, nyquist=mod))
    return np.concatenate(sets), ops, did


class TestRowAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = context()
        rng = cls.rng = np.random.default_rng(7)
        cls.axes, cls.bases, cls.iw, cls.n = setup(GRID, LO, HI)
        e = tuple(b.primal_extent for b in cls.bases)
        cls.C = rng.standard_normal(e + (5,)) * 0.4
        cls.cb = cls.ctx.buffer(4 * cls.n); cls.cb.upload(np.ascontiguousarray(cls.C.reshape(-1), np.float32))
        cls.vb = cls.ctx.buffer(4 * cls.n); cls.vb.upload(rng.standard_normal(cls.n).astype(np.float32))
        cls.o = [cls.ctx.buffer(4 * cls.n) for _ in range(3)]
        cls.lb = cls.ctx.buffer(4)
        cls.rows, cls.ops, cls.did = build(rng)
        cls.ops2, cls.plan = homogenize(cls.ops, absorb_s=["data"])
        cls.folded = fold(cls.rows, cls.ops2, cls.plan)

    def oracle(self, r, ops, tau):
        return row_loss_oracle(self.bases, self.iw, self.axes.encode(r["x"]), self.C, ops, r["op"],
                               r["w"].astype(np.float64), r["s"].astype(np.float64),
                               r["c"].astype(np.float64), 1.0,
                               modulus=r["nyquist"].astype(np.float64), tau=tau)

    def gpu(self, r, ops, tau, tier):
        """Loss/grad/diag/hvp of the whole row set on one tier. The JIT tier is
        one term per operator, all accumulating into the same buffers."""
        for b in self.o:
            b.zero()
        self.lb.zero()
        xe = self.axes.encode(r["x"])
        if tier == "generic":
            parts = [(EqRowTerm(self.ctx, self.bases, self.iw, ops), slice(None), r["op"])]
        else:
            parts = [(gk.GeneratedRowTerm(self.ctx, self.bases, self.iw, ops, int(k)), r["op"] == k,
                      np.zeros(int((r["op"] == k).sum()), np.int32)) for k in np.unique(r["op"])]
        for t, m, op in parts:
            t.tau = tau
            # FULL payload: homogenize puts the data target on a high slot, and a
            # [:, :5] truncation drops it in a way ONLY the gradient reveals
            t.bind_batch(xe[m], op, r["w"][m], r["s"][m], r["c"][m], modulus=r["nyquist"][m])
            t.loss(self.cb, self.lb)
            t.accumulate(self.cb, self.o[0])
            t.accumulate_diag(self.cb, self.o[1])
            t.hvp(self.cb, self.vb, self.o[2])
        out = [float(self.lb.download(np.float32, 1)[0])]
        return out + [b.download(np.float32, self.n).astype(np.float64) for b in self.o]

    def test_homogenize_plan(self):
        for k in range(self.ops.n_ops):
            self.assertTrue(is_homogeneous(self.ops2, k), self.ops.names[k])
        k_data = self.did
        self.assertIsNone(self.plan[k_data]["gain"]); self.assertIsNotNone(self.plan[k_data]["target"])
        self.assertEqual(self.ops2.kernels[k_data], "", "a changed operator drops its kernel hint")
        self.assertEqual(float(self.folded["w"].min()), 1.0); self.assertEqual(float(self.folded["w"].max()), 1.0)
        self.assertEqual(float(np.abs(self.folded["s"]).max()), 0.0)
        wr = self.rows["nyquist"] > 0                     # the modulus scaled with the payload
        self.assertTrue(np.allclose(self.folded["nyquist"][wr],
                                    self.rows["nyquist"][wr] * np.sqrt(self.rows["w"][wr]), rtol=1e-6))

    def test_same_objective_oracle(self):
        a, b = self.oracle(self.rows, self.ops, 0.25), self.oracle(self.folded, self.ops2, 0.25)
        self.assertLess(abs(a - b) / abs(a), 1e-6, "fold changed the objective")

    def test_same_objective_gpu(self):
        tiers = ["generic"] + (["jit"] if has_compiler() else [])
        for tau in (0.25, -1.0):
            for tier in tiers:
                with self.subTest(tier=tier, tau=tau):
                    g0 = self.gpu(self.rows, self.ops, tau, tier); g1 = self.gpu(self.folded, self.ops2, tau, tier)
                    self.assertLess(abs(g1[0] - g0[0]) / abs(g0[0]), 1e-5)
                    for i, nm in zip((1, 2, 3), ("grad", "diag", "hvp")):
                        self.assertLess(rel(g1[i], g0[i]), 1e-4, nm)

    def test_reweight_after_folding(self):
        """Through the generic GPU referee (itself checked against the oracle
        above): loss AND gradient of the reweighted-before and the
        reweighted-after systems agree."""
        k = self.rng.uniform(0.3, 4.0, len(self.rows))
        pre = self.rows.copy(); pre["w"] = pre["w"] * k.astype(np.float32)
        post = reweight(self.folded, k)
        a, b = self.gpu(pre, self.ops, 0.25, "generic"), self.gpu(post, self.ops2, 0.25, "generic")
        self.assertLess(abs(a[0] - b[0]) / abs(a[0]), 1e-5, "folded form is write-hostile")
        self.assertLess(rel(b[1], a[1]), 1e-4)
        sub = self.rows["op"] == self.did                  # masked, scalar factor
        pre2 = self.rows.copy(); pre2["w"][sub] *= 2
        post2 = reweight(self.folded, 2.0, mask=sub)
        a, b = self.gpu(pre2, self.ops, 0.25, "generic"), self.gpu(post2, self.ops2, 0.25, "generic")
        self.assertLess(abs(a[0] - b[0]) / abs(a[0]), 1e-5)
        self.assertLess(rel(b[1], a[1]), 1e-4)

    def test_refusals(self):
        with self.assertRaises(ValueError):                # s with no target slot
            fold(self.rows, *homogenize(self.ops))
        with self.assertRaises(ValueError):                # non-homogeneous operator
            fold(self.rows, self.ops, {k: {"gain": None, "target": None} for k in range(self.ops.n_ops)})

    def test_gain_of(self):
        w = gain_of(self.folded, self.ops2, self.plan)
        phys = ~np.isin(self.rows["op"], [self.did, self.ops.names.index("cont-frame")])
        err = np.abs(w[phys] - self.rows["w"][phys]) / self.rows["w"][phys]
        self.assertLess(err.max(), 1e-6)
        self.assertTrue(np.isnan(w[~phys]).all(), "data and frame rows are inseparable")

    def test_hash(self):
        self.assertEqual(rows_hash(self.folded, self.ops2), rows_hash(fold(self.rows, self.ops2, self.plan), self.ops2))
        self.assertNotEqual(rows_hash(self.folded, self.ops2), rows_hash(self.rows, self.ops))


if __name__ == "__main__":
    unittest.main()
