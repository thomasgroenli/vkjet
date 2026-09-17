"""Minibatched rows: the full solve is the K=1 special case, not a parallel
path, and the buckets are a partition of the one row system.

  buckets   row_buckets is an exact partition, sizes differ by <= 1, K=1 identity
  alias     K=1: the bucket IS the full batch object, scale exactly 1, every
            dispatch the unwrapped one (gradient and hvp to atomic noise)
  partition forcing all K (tol=0) reproduces the full-batch gradient
  variance  a partial draw's error is consistent with the reported variance,
            and CG sees ONE operator (hvp replays the drawn set)
  finite    the norm test passes once the whole row set is drawn
  jit       the JIT dispatch path binds every tier with the same partition
  fit       fit_rows minibatch=0 == K=1 end to end; K=4 is a sane fit
"""
import unittest

import numpy as np

from vkjet import (Axes, OperatorTable, make_rows, data_operator, fit_rows, EqRowTerm,
                   SLOT_VAL, SLOT_DX, SLOT_DY, SLOT_DZ)
from vkjet.rowfit import _jit_terms
from vkjet.rowbatch import row_buckets, local_buckets, BatchedRowSystem
from vkjet.tests import context, rel

LO = np.array([0., 0., 0., 0.]); HI = np.array([1., .06, .06, .10]); GRID = (6, 6, 6, 10)
NC, ND = 3_000, 9_000


def build(rng):
    """Two physics operators (one quadratic) + directional data rows."""
    ops = OperatorTable()
    cont = ops.add_op("continuity", lin=[(SLOT_DX, 0, 1.0), (SLOT_DY, 1, 1.0), (SLOT_DZ, 2, 1.0)])
    adv = ops.add_op("advection-x", lin=[(SLOT_DX, 3, 1.0)],
                     quad=[(SLOT_VAL, 0, SLOT_DX, 0, 1.0), (SLOT_VAL, 1, SLOT_DY, 0, 1.0)])
    ops, did = data_operator(ops)
    colloc = rng.uniform(LO, HI, (NC, 4)).astype(np.float32)
    eq = np.concatenate([make_rows(colloc, np.full(NC, k, np.int32), np.full(NC, 0.5, np.float32),
                                   np.zeros(NC, np.float32)) for k in (cont, adv)])
    x = rng.uniform(LO, HI, (ND, 4)).astype(np.float32)
    c = np.zeros((ND, 5), np.float32); c[:, :3] = rng.standard_normal((ND, 3))
    dr = make_rows(x, np.full(ND, did, np.int32), rng.uniform(0.2, 5.0, ND).astype(np.float32),
                   rng.standard_normal(ND).astype(np.float32), c)
    return np.concatenate([eq, dr]), ops


class TestMinibatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = context()
        rng = cls.rng = np.random.default_rng(11)
        cls.rows, cls.ops = build(rng)
        cls.x, cls.op, cls.w, cls.s, cls.c = (np.ascontiguousarray(cls.rows[k])
                                              for k in ("x", "op", "w", "s", "c"))
        cls.N = len(cls.rows)
        cls.axes = Axes(LO, HI, GRID, periodic=(True, False, False, False)); cls.bases = cls.axes.bases()
        cls.iw = [GRID[k] / (HI - LO)[k] for k in range(4)]
        cls.n = int(np.prod([b.primal_extent for b in cls.bases])) * 5
        cls.xe = cls.axes.encode(cls.x)
        cls.cb = cls.ctx.buffer(4 * cls.n); cls.cb.upload((rng.standard_normal(cls.n) * 0.2).astype(np.float32))
        cls.vb = cls.ctx.buffer(4 * cls.n); cls.vb.upload(rng.standard_normal(cls.n).astype(np.float32))
        cls.g = cls.ctx.buffer(4 * cls.n)
        ref = cls.term(); ref.bind_batch(cls.xe, cls.op, cls.w, cls.s, cls.c)
        cls.g_full, cls.hv_full = cls.accum(ref), cls.hvp(ref)
        cls.eta = max(rel(cls.accum(ref), cls.g_full), rel(cls.hvp(ref), cls.hv_full))

    @classmethod
    def term(cls):
        return EqRowTerm(cls.ctx, cls.bases, cls.iw, cls.ops)

    @classmethod
    def dl(cls, buf): return buf.download(np.float32, cls.n).astype(np.float64)

    @classmethod
    def accum(cls, obj):
        cls.g.zero()
        for t in (obj if isinstance(obj, list) else [obj]): t.accumulate(cls.cb, cls.g)
        return cls.dl(cls.g)

    @classmethod
    def hvp(cls, obj):
        cls.g.zero()
        for t in (obj if isinstance(obj, list) else [obj]): t.hvp(cls.cb, cls.vb, cls.g)
        return cls.dl(cls.g)

    def bucketed(self, K, seed, tol, sys_seed=0):
        t = self.term(); t.bind_buckets(self.xe, self.op, self.w, self.s, self.c, row_buckets(self.N, K, seed=seed))
        return t, BatchedRowSystem(self.ctx, [t], self.n, tol=tol, seed=sys_seed)

    def test_row_buckets_partition(self):
        for K in (1, 4, 7, 13):
            bk = row_buckets(self.N, K, seed=3); sz = sorted(len(b) for b in bk)
            self.assertEqual(len(bk), K)
            self.assertTrue(np.array_equal(np.sort(np.concatenate(bk)), np.arange(self.N)))
            self.assertLessEqual(sz[-1] - sz[0], 1)
        self.assertTrue(np.array_equal(row_buckets(self.N, 1)[0], np.arange(self.N)))

    def test_k1_is_an_alias(self):
        t1, sys1 = self.bucketed(1, 0, 0.15)
        self.assertIs(t1.minibatches[0], t1._batch, "K=1 copied the batch")
        self.assertFalse(sys1.stochastic); self.assertEqual(sys1.K, 1)
        g1, hv1 = self.accum(sys1), self.hvp(sys1); tok = sys1._batch; self.accum(sys1)
        self.assertLessEqual(rel(g1, self.g_full), max(3 * self.eta, 1e-8))
        self.assertLessEqual(rel(hv1, self.hv_full), max(3 * self.eta, 1e-8))
        self.assertEqual(sys1._scale, 1.0); self.assertIs(sys1._batch, tok)

    def test_all_buckets_reproduce_full_batch(self):
        for K in (4, 13):
            with self.subTest(K=K):
                tk, sk = self.bucketed(K, 5, 0.0)
                self.assertEqual(sum(b["n"] for b in tk.minibatches), tk._batch["n"])
                gk, hvk = self.accum(sk), self.hvp(sk)
                self.assertEqual(sk.eff_batch, (K, K)); self.assertEqual(sk._scale, 1.0)
                self.assertEqual(sk.grad_noise, 0.0)
                self.assertLess(rel(gk, self.g_full), 2e-6); self.assertLess(rel(hvk, self.hv_full), 2e-6)

    def test_partial_draw_variance_and_hvp_replay(self):
        K = 16
        tk, sk = self.bucketed(K, 9, 0.5, sys_seed=2)
        for trial in range(6):
            gk = self.accum(sk); used, sc = list(sk._used), sk._scale
            err = np.linalg.norm(gk - self.g_full)
            hvk = self.hvp(sk); man = np.zeros(self.n)
            for kk in used:
                self.g.zero(); tk.hvp(self.cb, self.vb, self.g, batch=tk.minibatches[kk]); man += self.dl(self.g)
            man *= sc
            self.assertLess(rel(hvk, man), 2e-6, trial)
            if len(used) == K:
                self.assertEqual(sk.grad_noise, 0.0); self.assertLess(err / np.linalg.norm(self.g_full), 1e-5)
            else:
                ratio = err / sk.grad_noise
                self.assertTrue(0.05 < ratio < 20.0, (trial, ratio))

    def test_finite_population_norm_test(self):
        tk2, s2 = self.bucketed(8, 1, 0.05, sys_seed=4); self.accum(s2)
        self.assertTrue(s2.batcher.passed); self.assertEqual(s2.eff_batch, (8, 8))
        self.assertEqual(s2.grad_noise, 0.0)

    def test_jit_dispatch_same_partition(self):
        of = np.empty(self.N, np.int32)
        for i, ix in enumerate(row_buckets(self.N, 8, seed=4)): of[ix] = i
        tt, handled = _jit_terms(self.ctx, self.axes, self.bases, GRID, self.x, self.op, self.w, self.s,
                                 self.c, self.ops, self.iw, False, fuzz=np.zeros(self.N, np.float32),
                                 modulus=np.zeros(self.N, np.float32), of_row=of, nb=8)
        terms = [t for t, _ in tt]
        if not handled.all():
            gen = self.term(); r = np.flatnonzero(~handled)
            gen.bind_buckets(self.axes.encode(self.x[r]), self.op[r], self.w[r], self.s[r], self.c[r],
                             local_buckets(of, r, 8)); terms.append(gen)
        sj = BatchedRowSystem(self.ctx, terms, self.n, tol=0.0)
        self.assertLess(rel(self.accum(sj), self.g_full), 5e-5)
        self.assertLess(rel(self.hvp(sj), self.hv_full), 5e-5)

    def test_fit_rows_k1_identity_and_k4(self):
        kw = dict(ops=self.ops, lo=LO, hi=HI, base_grid=GRID, n_stages=1, steps=6, cg_iters=6,
                  ctx=self.ctx, seed=1, verbose=False)
        r0 = fit_rows(self.rows, **kw, minibatch=0); r1 = fit_rows(self.rows, **kw, minibatch=self.N)
        dl = abs(r0.stage_losses[-1] - r1.stage_losses[-1]) / abs(r0.stage_losses[-1])
        dc = float(np.max(np.abs(r0.coef - r1.coef))) / max(float(np.max(np.abs(r0.coef))), 1e-30)
        self.assertLess(dl, 1e-4); self.assertLess(dc, 1e-3)
        r4 = fit_rows(self.rows, **kw, minibatch=self.N // 4, batch_tol=0.15)
        self.assertLess(r4.stage_losses[-1], r0.stage_losses[-1] * 1.5)


if __name__ == "__main__":
    unittest.main()
