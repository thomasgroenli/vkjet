"""Per-row jitter (`fuzz`): a row's evaluation point perturbed by N(0, σ²) in
CELL units, redrawn once per optimiser step. Exists because a FIXED collocation
set gets overfitted; a jittered row declares a measure, not a point.

  inert     σ = 0 changes nothing
  seed      the draw is a pure function of (row, seed): repeatable within a
            step, different between steps
  limit     σ -> 0 recovers the unfuzzed objective
  edge      rows at the domain boundary stay in-domain (no NaN)
  parity    generic and JIT compute the SAME jitter from the same seed
  grouped   the shared-gather kernel: σ = 0 matches generic, σ > 0 responds
  authoring make_rows carries the fuzz and nyquist columns through merge
"""
import unittest

import numpy as np

from vkjet.data import Axes, make_rows, merge_row_sets, data_operator
from vkjet.eqrow import EqRowTerm, OperatorTable, SLOT_VAL, SLOT_DX, SLOT_DY, SLOT_DZ
from vkjet.genkernel import GeneratedRowTerm, GroupedRowTerm
from vkjet.tests import context, needs_compiler

LO, HI, GRID = (0., 0., 0., 0.), (1., 1., 1., 1.), (8, 8, 8, 8)


def setup(ctx, n=4000, seed=0, edge=False):
    rng = np.random.default_rng(seed)
    axes = Axes(LO, HI, GRID, periodic=(True, False, False, False))
    bases = axes.bases()
    iw = [GRID[k] for k in range(4)]
    nco = int(np.prod([b.primal_extent for b in bases])) * 5
    ops = OperatorTable()
    ops.add_op("cont", lin=[(SLOT_DX, 0, 1.0), (SLOT_DY, 1, 1.0), (SLOT_DZ, 2, 1.0)])
    x = rng.uniform(0, 1, (n, 4)).astype(np.float32)
    if edge:                       # everything hard against the boundary
        x[:] = rng.choice([1e-5, 1 - 1e-5], size=(n, 4)).astype(np.float32)
    xe = np.ascontiguousarray(axes.encode(x), np.float32)
    coef = rng.standard_normal(nco).astype(np.float32) * 0.1
    cb = ctx.buffer(4 * nco); cb.upload(coef)
    return dict(axes=axes, bases=bases, iw=iw, ops=ops, xe=xe, op=np.zeros(n, np.int32),
                w=np.ones(n, np.float32), s=np.zeros(n, np.float32),
                c=np.zeros((n, 5), np.float32), cb=cb, lb=ctx.buffer(4), nco=nco)


def loss(term, cb, lb):
    lb.zero(); term.loss(cb, lb)
    return float(lb.download(np.float32, 1)[0])


class TestFuzz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = context()
        cls.f = setup(cls.ctx)
        f = cls.f
        cls.n = len(f["xe"])
        cls.t = EqRowTerm(cls.ctx, f["bases"], f["iw"], f["ops"])
        cls.t.bind_batch(f["xe"], f["op"], f["w"], f["s"], f["c"])
        cls.base = loss(cls.t, f["cb"], f["lb"])

    def bind(self, term, sigma, seed=None):
        f = self.f
        term.bind_batch(f["xe"], f["op"], f["w"], f["s"], f["c"],
                        fuzz=np.full(self.n, sigma, np.float32))
        if seed is not None:
            term.set_seed(seed)
        return loss(term, f["cb"], f["lb"])

    def test_sigma_zero_inert(self):
        z = self.bind(self.t, 0.0)
        self.t.set_seed(12345)
        z2 = loss(self.t, self.f["cb"], self.f["lb"])
        self.assertLessEqual(abs(z - self.base), 1e-6 * max(self.base, 1e-12))
        self.assertLessEqual(abs(z2 - self.base), 1e-6 * max(self.base, 1e-12))

    def test_seed_determinism(self):
        """Not bit-equality: the reduction uses float atomics, so repeated
        evaluations differ at ~1e-7 regardless of fuzz. A repeated seed sits at
        that floor; a different seed moves the loss by orders of magnitude more."""
        f = self.f
        a1 = self.bind(self.t, 0.35, seed=7); a2 = loss(self.t, f["cb"], f["lb"])
        self.t.set_seed(8); b1 = loss(self.t, f["cb"], f["lb"])
        self.t.set_seed(7); a3 = loss(self.t, f["cb"], f["lb"])
        rep = max(abs(a2 - a1), abs(a3 - a1)) / max(a1, 1e-12)
        dif = abs(b1 - a1) / max(a1, 1e-12)
        self.assertLess(rep, 1e-5, f"same seed not reproducible: {rep:.2e}")
        self.assertGreater(dif, 100 * max(rep, 1e-7), f"seed had no effect: {dif:.2e}")

    def test_sigma_to_zero_limit(self):
        prev = None
        for sg in (0.2, 0.05, 0.01, 0.002):
            d = abs(self.bind(self.t, sg, seed=3) - self.base) / max(self.base, 1e-12)
            if prev is not None:
                self.assertLess(d, prev * 1.5, (sg, d, prev))
            prev = max(d, 1e-12)
        self.assertLess(prev, 5e-2)

    def test_boundary_rows_stay_in_domain(self):
        g = setup(self.ctx, 2000, 1, edge=True)
        t2 = EqRowTerm(self.ctx, g["bases"], g["iw"], g["ops"])
        t2.bind_batch(g["xe"], g["op"], g["w"], g["s"], g["c"],
                      fuzz=np.full(len(g["xe"]), 1.5, np.float32))
        t2.set_seed(5)
        self.assertTrue(np.isfinite(loss(t2, g["cb"], g["lb"])))

    @needs_compiler
    def test_generic_vs_jit_parity(self):
        f = self.f
        g = GeneratedRowTerm(self.ctx, f["bases"], f["iw"], f["ops"], 0)
        worst = 0.0
        for sd in (0, 1, 42):
            for sg in (0.0, 0.1, 0.4):
                lg = self.bind(self.t, sg, seed=sd)
                g.bind_batch(f["xe"], np.zeros(self.n, np.int32), f["w"], f["s"], f["c"],
                             fuzz=np.full(self.n, sg, np.float32))
                g.set_seed(sd)
                lj = loss(g, f["cb"], f["lb"])
                worst = max(worst, abs(lg - lj) / max(abs(lg), 1e-12))
        self.assertLess(worst, 2e-4, worst)

    @needs_compiler
    def test_grouped_per_point_sigma(self):
        """σ rides a trailing column of the W plane, so every operator at a
        point gets the same offset and the shared gather survives. Grouped
        draws eps per POINT, generic per ROW: they agree at σ = 0 and are two
        valid mollifications otherwise (the gate is finite + responsive)."""
        f = self.f; n = self.n
        ops2 = OperatorTable()
        ops2.add_op("cont", lin=[(SLOT_DX, 0, 1.0), (SLOT_DY, 1, 1.0), (SLOT_DZ, 2, 1.0)])
        ops2.add_op("dtu", lin=[(SLOT_VAL, 0, 1.0), (SLOT_DZ, 2, 0.5)])
        rng = np.random.default_rng(5)
        wg = rng.uniform(0.5, 1.5, (n, 2)).astype(np.float32)
        sg_t = np.zeros((n, 2), np.float32)
        base_g = None
        for sd in (0, 4):
            for sig in (0.0, 0.25):
                sgv = np.full(n, sig, np.float32)
                gp = GroupedRowTerm(self.ctx, f["bases"], f["iw"], ops2, [0, 1])
                gp.bind_points(f["xe"], wg, sg_t, sigma=sgv); gp.set_seed(sd)
                ref = EqRowTerm(self.ctx, f["bases"], f["iw"], ops2)
                ref.bind_batch(np.repeat(f["xe"], 2, axis=0), np.tile(np.arange(2, dtype=np.int32), n),
                               wg.reshape(-1), sg_t.reshape(-1), np.zeros((2 * n, 5), np.float32),
                               fuzz=np.repeat(sgv, 2))
                ref.set_seed(sd)
                lg, lr = loss(gp, f["cb"], f["lb"]), loss(ref, f["cb"], f["lb"])
                self.assertTrue(np.isfinite(lg), (sd, sig, lg))
                if sig == 0.0:
                    self.assertLess(abs(lg - lr) / max(abs(lr), 1e-12), 2e-4)
                    base_g = lg
                else:
                    self.assertGreater(abs(lg - base_g) / max(base_g, 1e-12), 1e-3,
                                       f"sigma={sig} had no effect on the grouped kernel")
                del gp, ref

    def test_make_rows_carries_fuzz(self):
        """The authoring entry point carries the column (it once silently dropped it)."""
        ops, did = data_operator(None)
        r = make_rows(np.zeros((3, 4), np.float32), np.zeros(3, np.int32), np.ones(3, np.float32),
                      np.zeros(3, np.float32), nyquist=1.5, fuzz=0.5)
        m, _ = merge_row_sets((r, ops))
        self.assertTrue(np.all(r["fuzz"] == 0.5) and np.all(m["fuzz"] == 0.5)
                        and np.all(m["nyquist"] == 1.5))


if __name__ == "__main__":
    unittest.main()
