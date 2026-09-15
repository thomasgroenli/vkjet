"""End-to-end: fit_rows recovers a known field, and JIT == generic exactly.

Three gates:
  1. the JIT tier and the pure-generic tier produce the same objective
     (execution is an implementation detail, never a semantic one)
  2. a divergence-free field fitted from directional measurements + a
     continuity constraint is recovered
  3. the forward evaluator round-trips the fitted coefficients

Run: python3 tests/test_fit_rows.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vkjet import (Axes, OperatorTable, make_rows, merge_row_sets,   # noqa: E402
                   data_operator, fit_rows, Context,
                   SLOT_DX, SLOT_DY, SLOT_DZ)

LO, HI = (0., 0., 0., 0.), (1., 1., 1., 1.)
GRID, STAGES = (4, 4, 4, 4), 2


def truth(x):
    """An ABC-type field: EXACTLY divergence-free, fully 3D, time-modulated.

    Each component depends only on the two coordinates it is not differentiated
    by, so du/dx = dv/dy = dw/dz = 0 identically and the continuity rows are
    consistent with the truth rather than fighting it. Half-period (pi, not
    2*pi) over the unit box keeps it comfortably resolved by the finest grid.
    """
    tt, xx, yy, zz = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    g = 1.0 + 0.3 * np.cos(2 * np.pi * tt)
    u = (np.sin(np.pi * zz) + np.cos(np.pi * yy)) * g
    v = (np.sin(np.pi * xx) + np.cos(np.pi * zz)) * g
    w = (np.sin(np.pi * yy) + np.cos(np.pi * xx)) * g
    return np.stack([u, v, w, np.zeros_like(u), np.zeros_like(u)], 1)


def build(seed=0, n_data=40000, n_col=8000):
    rng = np.random.default_rng(seed)
    xd = rng.uniform(0, 1, (n_data, 4)).astype(np.float32)
    U = truth(xd)
    # random unit directional projections: r = <dir, u> - s
    dirs = rng.standard_normal((n_data, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    cov = np.zeros((n_data, 5), np.float32)
    cov[:, :3] = dirs
    s = (U[:, :3] * dirs).sum(1).astype(np.float32)
    dops, did = data_operator()
    drows = make_rows(xd, np.full(n_data, did, np.int32),
                      np.ones(n_data, np.float32), s, cov)

    # continuity as an equation row: du/dx + dv/dy + dw/dz = 0
    pops = OperatorTable()
    cid = pops.add_op("continuity", lin=[(SLOT_DX, 0, 1.0), (SLOT_DY, 1, 1.0),
                                         (SLOT_DZ, 2, 1.0)])
    xc = rng.uniform(0, 1, (n_col, 4)).astype(np.float32)
    prows = make_rows(xc, np.full(n_col, cid, np.int32),
                      np.full(n_col, 0.1, np.float32), np.zeros(n_col, np.float32))
    return merge_row_sets((drows, dops), (prows, pops))


def main():
    ctx = Context()
    print("device:", ctx.device_name)
    rows, ops = build()
    print(f"{len(rows):,} rows, {ops.n_ops} operators")

    out = {}
    for mode in (True, False):
        res = fit_rows(rows, ops, lo=LO, hi=HI, base_grid=GRID,
                       n_stages=STAGES, steps=12, cg_iters=8,
                       dispatch=mode, ctx=ctx, verbose=(mode is True))
        out[mode] = res
        print(f"  dispatch={str(mode):5s} final loss {res.stage_losses[-1]:.8g}")

    # --- gate 1: JIT == generic ------------------------------------------- #
    a, b = out[True].stage_losses[-1], out[False].stage_losses[-1]
    rel = abs(a - b) / max(abs(b), 1e-12)
    assert rel < 1e-3, f"JIT vs generic objective differs: {a} vs {b} (rel {rel:.2e})"
    print(f"  JIT vs generic objective: rel {rel:.2e}  OK")

    # --- gate 2 + 3: recovery through the forward evaluator ---------------- #
    rng = np.random.default_rng(99)
    xq = rng.uniform(0.05, 0.95, (20000, 4)).astype(np.float32)
    Ut = truth(xq)
    for mode in (True, False):
        Uh = out[mode].forward(xq)
        assert Uh.shape == (len(xq), 5), Uh.shape
        cc = [float(np.corrcoef(Uh[:, k], Ut[:, k])[0, 1]) for k in range(3)]
        # normalised RMSE: an absolute threshold would be meaningless, since it
        # only has scale relative to the field it is measuring
        rmse = float(np.sqrt(((Uh[:, :3] - Ut[:, :3]) ** 2).mean()))
        nrmse = rmse / float(np.sqrt((Ut[:, :3] ** 2).mean()))
        print(f"  dispatch={str(mode):5s} recovery corr u/v/w = "
              f"{cc[0]:.4f}/{cc[1]:.4f}/{cc[2]:.4f}  nRMSE {nrmse:.4f}")
        assert min(cc) > 0.97, f"poor recovery: {cc}"
        assert nrmse < 0.15, f"nRMSE too high: {nrmse}"

    print("PASS")
    ctx.destroy()


if __name__ == "__main__":
    main()
