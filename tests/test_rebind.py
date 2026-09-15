"""Re-authoring without rebinding: when a callable `rows` hands fit_rows a new
row system, an operator whose rows are unchanged keeps its bound term and a
common weight factor becomes the term's scale. The objective is identical, so
the solve must match a run that rebinds everything, to float tolerance.

Three operators: A unchanged, B reweighted by a schedule, C redrawn each time.
Run: PYTHONPATH=~/projects/volkano python3 tests/test_rebind.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vkjet import (Axes, OperatorTable, make_rows, merge_row_sets, data_operator,   # noqa: E402
                   fit_rows, Context, SLOT_DX, SLOT_DY, SLOT_DZ)
import vkjet.rowfit as rf                                                          # noqa: E402

LO, HI = (0., 0., 0., 0.), (1., 1., 1., 1.)


def truth(x):
    t, X, Y, Z = x.T
    return np.stack([np.sin(2*np.pi*Y) * np.cos(2*np.pi*t), np.sin(2*np.pi*Z), np.sin(2*np.pi*X),
                     0*X, 0*X], 1)


def system(seed, factor):
    rng = np.random.default_rng(seed)
    xa = rng.uniform(0, 1, (4000, 4)).astype(np.float32)             # A: fixed data rows
    ua = truth(xa); d = rng.normal(size=(4000, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    c = np.zeros((4000, 5), np.float32); c[:, :3] = d
    ops, did = data_operator(None)
    A = make_rows(xa, np.full(4000, did, np.int32), np.ones(4000, np.float32),
                  np.einsum("ij,ij->i", ua[:, :3], d).astype(np.float32), c)
    pops = OperatorTable()
    cid = pops.add_op("continuity", lin=[(SLOT_DX, 0, 1.0), (SLOT_DY, 1, 1.0), (SLOT_DZ, 2, 1.0)])
    xb = np.random.default_rng(1).uniform(0, 1, (3000, 4)).astype(np.float32)   # B: fixed points, weight schedule
    B = make_rows(xb, np.full(3000, cid, np.int32), np.full(3000, 0.05 * factor, np.float32),
                  np.zeros(3000, np.float32))
    return (A, ops), (B, pops), cid


def run(rebind_all, steps=6, resample=2):
    ctx = Context()
    calls = [0]
    A_set, (B, pops), cid = system(0, 1.0)

    def rows_for(grid):
        calls[0] += 1
        f = 4.0 * 0.6 ** calls[0]                                      # B's schedule
        Bk = B.copy(); Bk["w"] = np.float32(0.05 * f)
        rng = np.random.default_rng(100 + calls[0])                    # C: redrawn every call
        xc = rng.uniform(0, 1, (1000, 4)).astype(np.float32)
        C = make_rows(xc, np.full(1000, cid, np.int32), np.full(1000, 0.02, np.float32),
                      np.zeros(1000, np.float32))
        # a distinct operator id for C so it is its own term
        cops = OperatorTable(); c2 = cops.add_op("continuity-redrawn",
                                                 lin=[(SLOT_DX, 0, 1.0), (SLOT_DY, 1, 1.0), (SLOT_DZ, 2, 1.0 + 1e-7)])
        C["op"] = c2
        return merge_row_sets(A_set, (Bk, pops), (C, cops))

    if rebind_all:
        orig = rf.hashlib.blake2b
        counter = [0]

        def salted(*a, **k):                       # every key unique -> everything rebinds
            counter[0] += 1
            h = orig(*a, **k); h.update(str(counter[0]).encode()); return h
        rf.hashlib.blake2b = salted
    try:
        res = fit_rows(rows_for, None, lo=LO, hi=HI, base_grid=(4, 4, 4, 4), n_stages=1,
                       steps=steps, cg_iters=8, resample_every=resample, ctx=ctx, verbose=False)
    finally:
        if rebind_all:
            rf.hashlib.blake2b = orig
    coef = res.coef.copy(); losses = list(res.stage_losses)
    ctx.destroy()
    return coef, losses


def test_rebind():
    c_diff, l_diff = run(False)
    c_all, l_all = run(True)
    rel = np.abs(c_diff - c_all).max() / max(np.abs(c_all).max(), 1e-30)
    print(f"  diff-path vs rebind-all: max |dcoef| rel {rel:.2e}, loss {l_diff[-1]:.6g} vs {l_all[-1]:.6g}")
    assert rel < 1e-4 and abs(l_diff[-1] - l_all[-1]) <= 1e-5 * abs(l_all[-1]) + 1e-6
    print("REBIND OK — kept/rescaled terms define the same objective as rebinding")


if __name__ == "__main__":
    test_rebind()
