"""Per-row jitter (`fuzz`): parity, determinism, and the sigma->0 limit.

fuzz perturbs a row's evaluation point by N(0, sigma^2) in CELL units, redrawn
once per optimiser step. It exists because a FIXED collocation set gets
overfitted: at the converged production fit the physics residual is 125x
smaller in loss at the penalised points than at held-out lumen voxels, so the
physics term is a pointwise constraint rather than an integrated one
(examples/quadrature_overfit_probe.py).

Gates:
  1. sigma = 0 changes nothing (the field is off by default and inert)
  2. generic and JIT kernels compute the SAME jitter from the same seed
  3. the draw is a pure function of (row, seed): repeatable within a step,
     different between steps — the property the line search and CG depend on
  4. sigma -> 0 recovers the unfuzzed objective
  5. rows at the domain edge stay in-domain (no NaN from a point jittered out)

Run: PYTHONPATH=~/projects/volkano:~/genspline python3 tests/test_fuzz.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vkjet.context import Context                              # noqa: E402
from vkjet.data import Axes                                    # noqa: E402
from vkjet.eqrow import (EqRowTerm, OperatorTable, SLOT_VAL,   # noqa: E402
                          SLOT_DX, SLOT_DY, SLOT_DZ)
from vkjet.genkernel import (GeneratedRowTerm, GroupedRowTerm,  # noqa: E402
                              find_compiler)

LO, HI, GRID = (0., 0., 0., 0.), (1., 1., 1., 1.), (8, 8, 8, 8)


def setup(ctx, n=4000, seed=0, edge=False):
    rng = np.random.default_rng(seed)
    axes = Axes(LO, HI, GRID, periodic=(True, False, False, False))
    bases = axes.bases()
    ext = np.ones(4)
    iw = [GRID[k] / ext[k] for k in range(4)]
    nco = int(np.prod([b.primal_extent for b in bases])) * 5
    ops = OperatorTable()
    ops.add_op("cont", lin=[(SLOT_DX, 0, 1.0), (SLOT_DY, 1, 1.0),
                            (SLOT_DZ, 2, 1.0)])
    x = rng.uniform(0, 1, (n, 4)).astype(np.float32)
    if edge:                       # push everything hard against the boundary
        x[:] = rng.choice([1e-5, 1 - 1e-5], size=(n, 4)).astype(np.float32)
    xe = np.ascontiguousarray(axes.encode(x), np.float32)
    op = np.zeros(n, np.int32)
    w = np.ones(n, np.float32)
    s = np.zeros(n, np.float32)
    c = np.zeros((n, 5), np.float32)
    coef = rng.standard_normal(nco).astype(np.float32) * 0.1
    cb = ctx.buffer(4 * nco); cb.upload(coef)
    lb = ctx.buffer(4)
    return axes, bases, iw, ops, xe, op, w, s, c, cb, lb, nco


def loss(term, cb, lb):
    lb.zero(); term.loss(cb, lb)
    return float(lb.download(np.float32, 1)[0])


def main():
    ctx = Context()
    print("device:", ctx.device_name)
    axes, bases, iw, ops, xe, op, w, s, c, cb, lb, nco = setup(ctx)
    n = len(xe)

    # --- 1. sigma = 0 is inert -------------------------------------------- #
    t = EqRowTerm(ctx, bases, iw, ops)
    t.bind_batch(xe, op, w, s, c)
    base = loss(t, cb, lb)
    t.bind_batch(xe, op, w, s, c, fuzz=np.zeros(n, np.float32))
    z = loss(t, cb, lb)
    t.set_seed(12345)
    z2 = loss(t, cb, lb)
    assert abs(z - base) <= 1e-6 * max(base, 1e-12), (base, z)
    assert abs(z2 - base) <= 1e-6 * max(base, 1e-12), (base, z2)
    print(f"  sigma=0 inert: {base:.8g} == {z:.8g} == {z2:.8g}  OK")

    # --- 3. determinism + seed sensitivity -------------------------------- #
    fz = np.full(n, 0.35, np.float32)
    t.bind_batch(xe, op, w, s, c, fuzz=fz)
    # NOT bit-equality: the loss reduction uses float atomics, so repeated
    # evaluations of the SAME objective differ at ~1e-7 regardless of fuzz.
    # The gate is that a repeated seed sits at that floor while a different
    # seed moves the loss by orders of magnitude more.
    t.set_seed(7); a1 = loss(t, cb, lb); a2 = loss(t, cb, lb)
    t.set_seed(8); b1 = loss(t, cb, lb)
    t.set_seed(7); a3 = loss(t, cb, lb)
    rep = max(abs(a2 - a1), abs(a3 - a1)) / max(a1, 1e-12)
    dif = abs(b1 - a1) / max(a1, 1e-12)
    assert rep < 1e-5, f"same seed not reproducible: {rep:.2e}"
    assert dif > 100 * max(rep, 1e-7), f"seed had no effect: {dif:.2e} vs {rep:.2e}"
    print(f"  seed 7 -> {a1:.8g} (repeat spread {rep:.1e}), "
          f"seed 8 -> {b1:.8g} (moved {dif:.1e})  OK")

    # --- 4. sigma -> 0 recovers the unfuzzed objective --------------------- #
    prev = None
    for sg in (0.2, 0.05, 0.01, 0.002):
        t.bind_batch(xe, op, w, s, c, fuzz=np.full(n, sg, np.float32))
        t.set_seed(3)
        d = abs(loss(t, cb, lb) - base) / max(base, 1e-12)
        assert prev is None or d < prev * 1.5, (sg, d, prev)
        prev = max(d, 1e-12)
        print(f"  sigma={sg:<6g} |L-L0|/L0 = {d:.3e}")
    assert prev < 5e-2, prev

    # --- 5. boundary rows stay in-domain ----------------------------------- #
    _, bs2, iw2, ops2, xe2, op2, w2, s2, c2, cb2, lb2, _ = setup(ctx, 2000, 1,
                                                                 edge=True)
    t2 = EqRowTerm(ctx, bs2, iw2, ops2)
    t2.bind_batch(xe2, op2, w2, s2, c2, fuzz=np.full(len(xe2), 1.5, np.float32))
    t2.set_seed(5)
    le = loss(t2, cb2, lb2)
    assert np.isfinite(le), le
    print(f"  boundary rows with sigma=1.5 cells: loss {le:.6g} finite  OK")

    # --- 2. generic vs JIT parity WITH fuzz -------------------------------- #
    comp = find_compiler()
    if comp is None:
        print("  no GLSL compiler — JIT parity SKIPPED")
    else:
        g = GeneratedRowTerm(ctx, bases, iw, ops, 0, compiler=comp)
        worst = 0.0
        for sd in (0, 1, 42):
            for sg in (0.0, 0.1, 0.4):
                f_ = np.full(n, sg, np.float32)
                t.bind_batch(xe, op, w, s, c, fuzz=f_); t.set_seed(sd)
                g.bind_batch(xe, np.zeros(n, np.int32), w, s, c, fuzz=f_)
                g.set_seed(sd)
                lg, lj = loss(t, cb, lb), loss(g, cb, lb)
                worst = max(worst, abs(lg - lj) / max(abs(lg), 1e-12))
        assert worst < 2e-4, worst
        print(f"  generic vs JIT with fuzz, 9 (seed,sigma) combos: "
              f"worst rel {worst:.2e}  OK")

    # --- 6. GROUPED (shared-gather) path with per-point sigma -------------- #
    # sigma rides a trailing column of the W plane, so every operator at a
    # point gets the SAME offset and the shared gather survives. Checked
    # against the generic kernel fed equivalent per-row sigmas.
    if comp is not None:
        ops2 = OperatorTable()
        ops2.add_op("cont", lin=[(SLOT_DX, 0, 1.0), (SLOT_DY, 1, 1.0),
                                 (SLOT_DZ, 2, 1.0)])
        ops2.add_op("dtu", lin=[(SLOT_VAL, 0, 1.0), (SLOT_DZ, 2, 0.5)])
        rng = np.random.default_rng(5)
        wg = rng.uniform(0.5, 1.5, (n, 2)).astype(np.float32)
        sg_t = np.zeros((n, 2), np.float32)
        worst2 = 0.0
        for sd in (0, 4):
            for sig in (0.0, 0.25):
                sgv = np.full(n, sig, np.float32)
                gp = GroupedRowTerm(ctx, bases, iw, ops2, [0, 1], compiler=comp)
                gp.bind_points(xe, wg, sg_t, sigma=sgv)
                gp.set_seed(sd)
                ref = EqRowTerm(ctx, bases, iw, ops2)
                ref.bind_batch(np.repeat(xe, 2, axis=0),
                               np.tile(np.arange(2, dtype=np.int32), n),
                               wg.reshape(-1), sg_t.reshape(-1),
                               np.zeros((2 * n, 5), np.float32),
                               fuzz=np.repeat(sgv, 2))
                ref.set_seed(sd)
                lg, lr = loss(gp, cb, lb), loss(ref, cb, lb)
                rel2 = abs(lg - lr) / max(abs(lr), 1e-12)
                assert np.isfinite(lg), (sd, sig, lg)
                if sig == 0.0:
                    # identical objective: must match to kernel tolerance
                    assert rel2 < 2e-4, f"grouped != generic at sigma=0: {rel2:.2e}"
                    base_g = lg
                else:
                    # grouped draws eps per POINT, generic per ROW, so the two
                    # are different (both valid) mollifications and need not
                    # agree — but fuzz must actually be doing something
                    assert abs(lg - base_g) / max(base_g, 1e-12) > 1e-3, (
                        f"sigma={sig} had no effect on the grouped kernel")
                worst2 = max(worst2, rel2)
                del gp, ref
        # NOTE: grouped indexes eps by POINT, generic by ROW, so the draws are
        # only expected to agree when sigma = 0. With sigma > 0 the two use
        # different (valid) mollifications; the gate is that grouped stays
        # finite and actually responds to sigma.
        print(f"  grouped: sigma=0 matches generic, sigma>0 responds "
              f"(spread across arms {worst2:.2e}, expected — per-point vs "
              f"per-row eps)")

    print("PASS")
    ctx.destroy()


if __name__ == "__main__":
    main()
