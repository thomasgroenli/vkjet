"""JIT kernel generation: parity with the generic referee + cache integrity.

All FOUR generated kernels (loss, grad, diag, hvp), the boundary cells of
every dimension, the cache-key ≡ emitted-code invariant, rejection of damaged
cache entries, degenerate operators, and that verify_generated catches a
defect in any kernel and is memoised across a ladder.
"""
import os
import unittest

import numpy as np

from vkjet.eqrow import OperatorTable, EqRowTerm, NCPR, SLOT_VAL, SLOT_DX
from vkjet import genkernel as gk
from vkjet.tests import context, needs_compiler, scratch_dir
from vkjet.tests._fixtures import setup, sample_operators, four_way

RTOL = 2e-4


@needs_compiler
class TestGenKernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = context()
        cls.cache = scratch_dir()
        gk._VERIFIED.clear()

    def test_per_op_parity(self):
        """Every generated single-op kernel matches the generic term on all four."""
        axes, bases, iw, nco = setup()
        ops, ids = sample_operators()
        rng = np.random.default_rng(0)
        n = 96
        x = gk.verify_points(bases, rng, n)
        w = rng.uniform(.5, 1.5, n).astype(np.float32)
        s = (rng.standard_normal(n) * .1).astype(np.float32)
        c = rng.standard_normal((n, NCPR)).astype(np.float32)
        for name, k in ids.items():
            with self.subTest(op=name):
                sub = OperatorTable(); sub.add_op(ops.names[k], ops.lin[k], ops.quad[k])
                ref = EqRowTerm(self.ctx, bases, iw, sub)
                ref.bind_batch(x, np.zeros(n, np.int32), w, s, c)
                gt = gk.GeneratedRowTerm(self.ctx, bases, iw, ops, k, cache_dir=self.cache)
                gt.bind_batch(x, np.zeros(n, np.int32), w, s, c)
                r = four_way(self.ctx, ref, gt, nco)
                self.assertLess(max(r.values()), RTOL, f"{name}: {r}")

    def test_grouped_parity(self):
        """The shared-gather kernel matches m independent generic rows."""
        axes, bases, iw, nco = setup()
        ops, ids = sample_operators()
        gids = [ids["mom0"], ids["mom1"], ids["mom2"], ids["cont"], ids["of"]]
        m = len(gids)
        rng = np.random.default_rng(2)
        n = 64
        xp = gk.verify_points(bases, rng, n)
        W = rng.uniform(.5, 1.5, (n, m)).astype(np.float32)
        S = (rng.standard_normal((n, m)) * .05).astype(np.float32)
        sub = OperatorTable()
        for k in gids:
            sub.add_op(ops.names[k], ops.lin[k], ops.quad[k])
        ref = EqRowTerm(self.ctx, bases, iw, sub)
        ref.bind_batch(np.repeat(xp, m, axis=0), np.tile(np.arange(m, dtype=np.int32), n),
                       W.reshape(-1), S.reshape(-1), None)
        gt = gk.GroupedRowTerm(self.ctx, bases, iw, ops, gids, cache_dir=self.cache)
        gt.bind_points(xp, W, S)
        r = four_way(self.ctx, ref, gt, nco)
        self.assertLess(max(r.values()), RTOL, r)

    def test_cache_key_is_the_code(self):
        """Two operators may share a cache entry ONLY if their GLSL is identical.
        Regression: structure_hash once stripped values while emit_shaders
        sorted BY value, so two ops with one hash emitted different shaders."""
        axes, bases, iw, nco = setup()
        a = OperatorTable(); a.add_op("A", lin=[(SLOT_VAL, 0, 5.0, 1), (SLOT_VAL, 0, 1.0, 2)])
        b = OperatorTable(); b.add_op("B", lin=[(SLOT_VAL, 0, 1.0, 1), (SLOT_VAL, 0, 5.0, 2)])
        sa, va = gk.emit_shaders(a, 0)
        sb, vb_ = gk.emit_shaders(b, 0)
        self.assertEqual(sa, sb, "same structure must emit identical GLSL")
        self.assertEqual(gk.source_key(sa), gk.source_key(sb))
        self.assertFalse(np.allclose(va, vb_), "the values buffer distinguishes them")
        rng = np.random.default_rng(5)
        n = 64
        x = gk.verify_points(bases, rng, n)
        w = np.ones(n, np.float32)
        s = (rng.standard_normal(n) * .1).astype(np.float32)
        c = rng.standard_normal((n, NCPR)).astype(np.float32)
        ref = EqRowTerm(self.ctx, bases, iw, b); ref.bind_batch(x, np.zeros(n, np.int32), w, s, c)
        gk.GeneratedRowTerm(self.ctx, bases, iw, a, 0, cache_dir=self.cache)     # populates
        tb = gk.GeneratedRowTerm(self.ctx, bases, iw, b, 0, cache_dir=self.cache)  # reuses
        tb.bind_batch(x, np.zeros(n, np.int32), w, s, c)
        r = four_way(self.ctx, ref, tb, nco)
        self.assertLess(max(r.values()), RTOL, f"shared-cache aliasing: {r}")

    def test_damaged_cache_rejected(self):
        """A truncated/garbage .spv is recompiled, not trusted."""
        axes, bases, iw, _ = setup()
        ops, ids = sample_operators()
        d = scratch_dir()
        t = gk.GeneratedRowTerm(self.ctx, bases, iw, ops, ids["cont"], cache_dir=d)
        p = os.path.join(d, f"{t.cache_key}_hvp.spv")
        full = os.path.getsize(p)
        with open(p, "r+b") as f:                       # a killed compile
            f.truncate(full // 2)
        self.assertFalse(gk.valid_spv(p))
        gk.GeneratedRowTerm(self.ctx, bases, iw, ops, ids["cont"], cache_dir=d)
        self.assertEqual(os.path.getsize(p), full, "cache must be repaired")
        with open(p, "r+b") as f:
            f.write(b"junk")
        self.assertFalse(gk.valid_spv(p), "bad magic must be rejected")

    def test_degenerate_operators(self):
        """Entry-free operators refuse cleanly; a group with one empty member
        still compiles and is exact; a payload op is refused by the grouped
        emitter."""
        ops = OperatorTable()
        empty = ops.add_op("empty")
        real = ops.add_op("real", lin=[(SLOT_DX, 0, 1.0)])
        with self.assertRaises(ValueError):
            gk.emit_shaders(ops, empty)
        with self.assertRaises(ValueError):
            gk.emit_group_shaders(ops, [empty])
        srcs, _ = gk.emit_group_shaders(ops, [real, empty])
        self.assertNotIn("+= ;", srcs["hvp"])
        axes, bases, iw, nco = setup()
        rng = np.random.default_rng(9)
        n, m = 48, 2
        xp = gk.verify_points(bases, rng, n)
        W = rng.uniform(.5, 1.5, (n, m)).astype(np.float32)
        S = (rng.standard_normal((n, m)) * .05).astype(np.float32)
        sub = OperatorTable()
        for k in (real, empty):
            sub.add_op(ops.names[k], ops.lin[k], ops.quad[k])
        ref = EqRowTerm(self.ctx, bases, iw, sub)
        ref.bind_batch(np.repeat(xp, m, axis=0), np.tile(np.arange(m, dtype=np.int32), n),
                       W.reshape(-1), S.reshape(-1), None)
        gt = gk.GroupedRowTerm(self.ctx, bases, iw, ops, [real, empty], cache_dir=self.cache)
        gt.bind_points(xp, W, S)
        r = four_way(self.ctx, ref, gt, nco)
        self.assertLess(max(r.values()), RTOL, f"empty group member: {r}")
        ops2 = OperatorTable(); ops2.add_op("pay", lin=[(SLOT_VAL, 0, 1.0, 3)])
        with self.assertRaises(ValueError):
            gk.emit_group_shaders(ops2, [0])

    def test_verify_catches_sabotage(self):
        """verify_generated fails on a defect in ANY kernel, including one
        confined to the last cell of a dimension."""
        axes, bases, iw, _ = setup()
        ops, ids = sample_operators()
        k = ids["mom0"]
        real = gk.emit_shaders

        def sabotage(kind, find, repl):
            def patched(o, kk):
                srcs, vals = real(o, kk)
                self.assertIn(find, srcs[kind])
                srcs = dict(srcs); srcs[kind] = srcs[kind].replace(find, repl)
                return srcs, vals
            return patched

        cases = [("diag", "float sw = meta.scale * rw;", "float sw = 2.0 * meta.scale * rw;"),
                 ("hvp", "float y = meta.scale * rw * Jv;", "float y = 2.0 * meta.scale * rw * Jv;"),
                 ("grad", "float a = meta.scale * rw * softwrap(r, rm, meta.tau);",
                  "float a = meta.scale * rw * softwrap(r, rm, meta.tau); "
                  "if (ii[3] >= meta.primal_extent[3]-1) return;")]
        for kind, find, repl in cases:
            with self.subTest(kernel=kind):
                gk.emit_shaders = sabotage(kind, find, repl)
                try:
                    t = gk.GeneratedRowTerm(self.ctx, bases, iw, ops, k, cache_dir=scratch_dir())
                    with self.assertRaises(RuntimeError) as cm:
                        gk.verify_generated(self.ctx, t, bases, iw, ops, k, force=True)
                    self.assertIn("parity failed", str(cm.exception))
                finally:
                    gk.emit_shaders = real

    def test_verify_memoized(self):
        """The ladder verifies once, not once per stage."""
        ops, ids = sample_operators()
        calls = []
        for grid in ((4, 4, 4, 6), (8, 8, 8, 12), (16, 16, 16, 24)):
            axes, bases, iw, _ = setup(grid)
            t = gk.GeneratedRowTerm(self.ctx, bases, iw, ops, ids["cont"], cache_dir=self.cache)
            before = len(gk._VERIFIED)
            gk.verify_generated(self.ctx, t, bases, iw, ops, ids["cont"])
            calls.append(len(gk._VERIFIED) - before)
        self.assertEqual(calls[1:], [0, 0], f"re-verified at finer stages: {calls}")


if __name__ == "__main__":
    unittest.main()
