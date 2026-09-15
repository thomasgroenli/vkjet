"""JIT kernel generation: parity with the generic referee + cache integrity.

Covers what the audit found missing: all FOUR generated kernels (loss, grad,
diag, hvp) rather than loss+grad, the boundary cells of every dimension, the
cache-key ≡ emitted-code invariant, and rejection of damaged cache entries.

Run: python3 tests/test_genkernel.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vkjet.context import Context                                  # noqa: E402
from vkjet.data import Axes                                        # noqa: E402
from vkjet.eqrow import (OperatorTable, EqRowTerm, NCH, NCPR,      # noqa: E402
                          SLOT_VAL, SLOT_DT, SLOT_DX, SLOT_DY, SLOT_DZ,
                          SLOT_DXX, SLOT_DYY, SLOT_DZZ,
                          SLOT_DTX, SLOT_DYZ)
from vkjet import genkernel as gk                                  # noqa: E402


LO, HI, GRID = (0., 0., 0., 0.), (1., 6., 6., 10.), (4, 4, 4, 6)
RTOL = 2e-4


def _setup(ctx, grid=GRID):
    axes = Axes(LO, HI, grid, periodic=(True, False, False, False))
    bases = axes.bases()
    iw = [grid[k] / (HI[k] - LO[k]) for k in range(4)]
    nco = int(np.prod([b.primal_extent for b in bases])) * NCH
    return axes, bases, iw, nco


def _four_way(ctx, ref, term, nco, seed=1):
    """→ dict of relative errors for loss/grad/diag/hvp (terms already bound)."""
    rng = np.random.default_rng(seed)
    cb = ctx.buffer(nco * 4); cb.upload((rng.standard_normal(nco) * .3).astype(np.float32))
    vb = ctx.buffer(nco * 4); vb.upload((rng.standard_normal(nco) * .3).astype(np.float32))
    lb = ctx.buffer(4); ab = ctx.buffer(nco * 4)
    out = []
    for t in (ref, term):
        lb.zero(); t.loss(cb, lb)
        res = [float(lb.download(np.float32, 1)[0])]
        for run in (lambda: t.accumulate(cb, ab),
                    lambda: t.accumulate_diag(cb, ab),
                    lambda: t.hvp(cb, vb, ab)):
            ab.zero(); run(); res.append(ab.download(np.float32, nco))
        out.append(res)
    a, b = out
    rel = {"loss": abs(b[0] - a[0]) / max(abs(a[0]), 1e-9)}
    for name, x0, x1 in zip(("grad", "diag", "hvp"), a[1:], b[1:]):
        rel[name] = float(np.linalg.norm(x1 - x0) / max(np.linalg.norm(x0), 1e-9))
    return rel


# --------------------------------------------------------------------------- #
def sample_operators():
    """A generic operator set spanning the shapes the emitter must handle:
    linear-only, quadratic, mixed second derivatives, and a payload (cix) op.
    Stands in for any application's physics — vkjet itself is domain-free."""
    ops, ids = OperatorTable(), {}
    for i in range(3):                      # advection-shaped: lin + quad
        sp = (SLOT_DX, SLOT_DY, SLOT_DZ)
        lin = [(SLOT_DT, i, 1.0), (sp[i], 3, 1.0)]
        lin += [(dd, i, -1e-3) for dd in (SLOT_DXX, SLOT_DYY, SLOT_DZZ)]
        quad = [(SLOT_VAL, k, sp[k], i, 1.0) for k in range(3)]
        quad += [(SLOT_VAL, i, sp[k], k, 0.5) for k in range(3)]
        ids[f"mom{i}"] = ops.add_op(f"mom-{i}", lin, quad)
    ids["cont"] = ops.add_op("continuity",
                             lin=[(SLOT_DX, 0, 1.0), (SLOT_DY, 1, 1.0),
                                  (SLOT_DZ, 2, 1.0)])
    ids["of"] = ops.add_op("transport", lin=[(SLOT_DT, 4, 1.0)],
                           quad=[(SLOT_VAL, k, (SLOT_DX, SLOT_DY, SLOT_DZ)[k], 4, 1.0)
                                 for k in range(3)])
    return ops, ids


def test_per_op_parity(ctx, cache):
    """Every generated single-op kernel matches the generic term on all four."""
    axes, bases, iw, nco = _setup(ctx)
    ops, ids = sample_operators()
    # + a payload-carrying operator and one touching second/mixed derivatives
    ids["data"] = ops.add_op("data", lin=[(SLOT_VAL, c, 1.0, c) for c in range(5)])
    ids["mixed"] = ops.add_op(
        "mixed", lin=[(SLOT_DXX, 1, .7), (SLOT_DTX, 0, -.3), (SLOT_DYZ, 4, .2)],
        quad=[(SLOT_DY, 2, SLOT_DY, 2, .5), (SLOT_VAL, 3, SLOT_DZ, 3, -.4)])
    rng = np.random.default_rng(0)
    n = 96
    x = gk.verify_points(bases, rng, n)
    w = rng.uniform(.5, 1.5, n).astype(np.float32)
    s = (rng.standard_normal(n) * .1).astype(np.float32)
    c = rng.standard_normal((n, NCPR)).astype(np.float32)
    worst = {}
    for name, k in ids.items():
        sub = OperatorTable(); sub.add_op(ops.names[k], ops.lin[k], ops.quad[k])
        ref = EqRowTerm(ctx, bases, iw, sub)
        ref.bind_batch(x, np.zeros(n, np.int32), w, s, c)
        gt = gk.GeneratedRowTerm(ctx, bases, iw, ops, k, cache_dir=cache)
        gt.bind_batch(x, np.zeros(n, np.int32), w, s, c)
        rel = _four_way(ctx, ref, gt, nco)
        assert max(rel.values()) < RTOL, f"{name}: {rel}"
        for kk, v in rel.items():
            worst[kk] = max(worst.get(kk, 0.0), v)
    print("  per-op parity (8 ops, 4 kernels) worst:",
          {k: f"{v:.1e}" for k, v in worst.items()})


def test_grouped_parity(ctx, cache):
    """The shared-gather kernel matches m independent generic rows."""
    axes, bases, iw, nco = _setup(ctx)
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
    ref = EqRowTerm(ctx, bases, iw, sub)
    ref.bind_batch(np.repeat(xp, m, axis=0), np.tile(np.arange(m, dtype=np.int32), n),
                   W.reshape(-1), S.reshape(-1), None)
    gt = gk.GroupedRowTerm(ctx, bases, iw, ops, gids, cache_dir=cache)
    gt.bind_points(xp, W, S)
    rel = _four_way(ctx, ref, gt, nco)
    assert max(rel.values()) < RTOL, rel
    print("  grouped parity (m=5, 4 kernels):", {k: f"{v:.1e}" for k, v in rel.items()})


def test_cache_key_is_the_code(ctx, cache):
    """Two operators may share a cache entry ONLY if their GLSL is identical.

    Regression: structure_hash used to strip values while emit_shaders sorted
    BY value, so two ops with the same hash could emit different shaders and
    the second silently ran the first's kernel (~80% gradient error)."""
    axes, bases, iw, nco = _setup(ctx)
    a = OperatorTable(); a.add_op("A", lin=[(SLOT_VAL, 0, 5.0, 1), (SLOT_VAL, 0, 1.0, 2)])
    b = OperatorTable(); b.add_op("B", lin=[(SLOT_VAL, 0, 1.0, 1), (SLOT_VAL, 0, 5.0, 2)])
    sa, va = gk.emit_shaders(a, 0)
    sb, vb_ = gk.emit_shaders(b, 0)
    # emission is now a function of the INDEX structure alone
    assert sa == sb, "same structure must emit identical GLSL"
    assert gk.source_key(sa) == gk.source_key(sb)
    assert not np.allclose(va, vb_), "the values buffer is what distinguishes them"

    # and the shared cache must give B the same answer as a clean cache
    rng = np.random.default_rng(5)
    n = 64
    x = gk.verify_points(bases, rng, n)
    w = np.ones(n, np.float32)
    s = (rng.standard_normal(n) * .1).astype(np.float32)
    c = rng.standard_normal((n, NCPR)).astype(np.float32)
    ref = EqRowTerm(ctx, bases, iw, b); ref.bind_batch(x, np.zeros(n, np.int32), w, s, c)
    ta = gk.GeneratedRowTerm(ctx, bases, iw, a, 0, cache_dir=cache)   # populates
    tb = gk.GeneratedRowTerm(ctx, bases, iw, b, 0, cache_dir=cache)   # reuses
    tb.bind_batch(x, np.zeros(n, np.int32), w, s, c)
    rel = _four_way(ctx, ref, tb, nco)
    assert max(rel.values()) < RTOL, f"shared-cache aliasing: {rel}"
    print("  cache-key ≡ code: shared cache exact,",
          {k: f"{v:.1e}" for k, v in rel.items()})


def test_damaged_cache_rejected(ctx, cache):
    """A truncated/garbage .spv must be recompiled, not trusted."""
    axes, bases, iw, _ = _setup(ctx)
    ops, ids = sample_operators()
    d = tempfile.mkdtemp()
    t = gk.GeneratedRowTerm(ctx, bases, iw, ops, ids["cont"], cache_dir=d)
    p = os.path.join(d, f"{t.cache_key}_hvp.spv")
    full = os.path.getsize(p)
    with open(p, "r+b") as f:                       # simulate a killed compile
        f.truncate(full // 2)
    assert not gk.valid_spv(p), "truncated module must be rejected"
    gk.GeneratedRowTerm(ctx, bases, iw, ops, ids["cont"], cache_dir=d)
    assert os.path.getsize(p) == full, "cache must be repaired, not reused"
    with open(p, "r+b") as f:
        f.write(b"junk")
    assert not gk.valid_spv(p), "bad magic must be rejected"
    print(f"  damaged cache rejected + repaired ({full} bytes)")


def test_degenerate_operators(ctx, cache):
    """Entry-free operators refuse cleanly instead of emitting invalid GLSL."""
    ops = OperatorTable()
    empty = ops.add_op("empty")
    real = ops.add_op("real", lin=[(SLOT_DX, 0, 1.0)])
    for fn, args in ((gk.emit_shaders, (ops, empty)),
                     (gk.emit_group_shaders, (ops, [empty]))):
        try:
            fn(*args)
            raise AssertionError(f"{fn.__name__} should refuse an empty operator")
        except ValueError:
            pass
    # a group with ONE empty member must still compile AND be exact — it used
    # to emit "Jv1 += ;" and take the whole group down with it
    srcs, _ = gk.emit_group_shaders(ops, [real, empty])
    assert "+= ;" not in srcs["hvp"]
    axes, bases, iw, nco = _setup(ctx)
    rng = np.random.default_rng(9)
    n, m = 48, 2
    xp = gk.verify_points(bases, rng, n)
    W = rng.uniform(.5, 1.5, (n, m)).astype(np.float32)
    S = (rng.standard_normal((n, m)) * .05).astype(np.float32)
    sub = OperatorTable()
    for k in (real, empty):
        sub.add_op(ops.names[k], ops.lin[k], ops.quad[k])
    ref = EqRowTerm(ctx, bases, iw, sub)
    ref.bind_batch(np.repeat(xp, m, axis=0), np.tile(np.arange(m, dtype=np.int32), n),
                   W.reshape(-1), S.reshape(-1), None)
    gt = gk.GroupedRowTerm(ctx, bases, iw, ops, [real, empty], cache_dir=cache)
    gt.bind_points(xp, W, S)
    rel = _four_way(ctx, ref, gt, nco)
    assert max(rel.values()) < RTOL, f"empty group member: {rel}"
    ops2 = OperatorTable(); ops2.add_op("pay", lin=[(SLOT_VAL, 0, 1.0, 3)])
    try:
        gk.emit_group_shaders(ops2, [0])
        raise AssertionError("payload op must be refused by the grouped emitter")
    except ValueError:
        pass
    print("  degenerate shapes refused with ValueError (survives python -O)")


def test_verify_catches_sabotage(ctx, cache):
    """verify_generated must fail on a defect in ANY of the four kernels,
    including one confined to the last cell of a dimension."""
    axes, bases, iw, _ = _setup(ctx)
    ops, ids = sample_operators()
    k = ids["mom0"]
    real = gk.emit_shaders

    def sabotage(kind, find, repl):
        def patched(o, kk):
            srcs, vals = real(o, kk)
            assert find in srcs[kind], f"pattern not in {kind}"
            srcs = dict(srcs); srcs[kind] = srcs[kind].replace(find, repl)
            return srcs, vals
        return patched

    cases = [("diag", "float sw = meta.scale * rw;", "float sw = 2.0 * meta.scale * rw;"),
             ("hvp", "float y = meta.scale * rw * Jv;", "float y = 2.0 * meta.scale * rw * Jv;"),
             ("grad", "float a = meta.scale * rw * softwrap(r, rm, meta.tau);",
              "float a = meta.scale * rw * softwrap(r, rm, meta.tau); if (ii[3] >= meta.primal_extent[3]-1) return;")]
    for kind, find, repl in cases:
        d = tempfile.mkdtemp()
        gk.emit_shaders = sabotage(kind, find, repl)
        try:
            t = gk.GeneratedRowTerm(ctx, bases, iw, ops, k, cache_dir=d)
            try:
                gk.verify_generated(ctx, t, bases, iw, ops, k, force=True)
                raise AssertionError(f"verify_generated MISSED a {kind} defect")
            except RuntimeError as ex:
                assert "parity failed" in str(ex), ex
        finally:
            gk.emit_shaders = real
    print("  verify caught sabotage in diag, hvp, and a last-cell-only grad defect")


def test_verify_memoized(ctx, cache):
    """The ladder verifies once, not once per stage."""
    ops, ids = sample_operators()
    calls = []
    for grid in ((4, 4, 4, 6), (8, 8, 8, 12), (16, 16, 16, 24)):
        axes, bases, iw, _ = _setup(ctx, grid)
        t = gk.GeneratedRowTerm(ctx, bases, iw, ops, ids["cont"], cache_dir=cache)
        before = len(gk._VERIFIED)
        gk.verify_generated(ctx, t, bases, iw, ops, ids["cont"])
        calls.append(len(gk._VERIFIED) - before)
    assert calls[1:] == [0, 0], f"re-verified at finer stages: {calls}"
    print(f"  verification memoized across the ladder (new entries per stage: {calls})")


def main():
    if gk.find_compiler() is None:
        print("no glslc/glslangValidator — skipping"); return
    ctx = Context()
    print("device:", ctx.device_name)
    cache = tempfile.mkdtemp()
    gk._VERIFIED.clear()
    for fn in (test_per_op_parity, test_grouped_parity, test_cache_key_is_the_code,
               test_damaged_cache_rejected, test_degenerate_operators,
               test_verify_catches_sabotage, test_verify_memoized):
        print(f"{fn.__name__}:")
        fn(ctx, cache)
    print("\nALL GENKERNEL TESTS PASSED")


if __name__ == "__main__":
    main()
