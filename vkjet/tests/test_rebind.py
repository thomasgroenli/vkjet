"""Re-authoring without rebinding: when a callable `rows` hands fit_rows a new
row system, an operator whose rows are unchanged keeps its bound term and a
common weight factor becomes the term's scale. The objective is identical, so
the solve must match a run that rebinds everything, to float tolerance.

Three operators: A unchanged, B reweighted by a schedule, C redrawn each time.
"""
import unittest

import numpy as np

from vkjet import (OperatorTable, make_rows, merge_row_sets, data_operator, fit_rows,
                   SLOT_DX, SLOT_DY, SLOT_DZ)
import vkjet.rowfit as rf
from vkjet.tests import context

LO, HI = (0., 0., 0., 0.), (1., 1., 1., 1.)


def truth(x):
    t, X, Y, Z = x.T
    return np.stack([np.sin(2*np.pi*Y) * np.cos(2*np.pi*t), np.sin(2*np.pi*Z), np.sin(2*np.pi*X),
                     0*X, 0*X], 1)


def system(seed):
    rng = np.random.default_rng(seed)
    n = 4000
    xa = rng.uniform(0, 1, (n, 4)).astype(np.float32)               # A: fixed data rows
    ua = truth(xa); d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    c = np.zeros((n, 5), np.float32); c[:, :3] = d
    ops, did = data_operator(None)
    A = make_rows(xa, np.full(n, did, np.int32), np.ones(n, np.float32),
                  np.einsum("ij,ij->i", ua[:, :3], d).astype(np.float32), c)
    pops = OperatorTable()
    cid = pops.add_op("continuity", lin=[(SLOT_DX, 0, 1.0), (SLOT_DY, 1, 1.0), (SLOT_DZ, 2, 1.0)])
    xb = np.random.default_rng(1).uniform(0, 1, (3000, 4)).astype(np.float32)   # B: weight schedule
    B = make_rows(xb, np.full(3000, cid, np.int32), np.full(3000, 0.05, np.float32),
                  np.zeros(3000, np.float32))
    return (A, ops), (B, pops), cid


def run(ctx, rebind_all, steps=6, resample=2):
    calls = [0]
    A_set, (B, pops), cid = system(0)

    def rows_for(grid):
        calls[0] += 1
        Bk = B.copy(); Bk["w"] = np.float32(0.05 * 4.0 * 0.6 ** calls[0])   # B's schedule
        rng = np.random.default_rng(100 + calls[0])                          # C: redrawn every call
        xc = rng.uniform(0, 1, (1000, 4)).astype(np.float32)
        C = make_rows(xc, np.full(1000, cid, np.int32), np.full(1000, 0.02, np.float32),
                      np.zeros(1000, np.float32))
        cops = OperatorTable()               # a distinct operator so C is its own term
        C["op"] = cops.add_op("continuity-redrawn",
                              lin=[(SLOT_DX, 0, 1.0), (SLOT_DY, 1, 1.0), (SLOT_DZ, 2, 1.0 + 1e-7)])
        return merge_row_sets(A_set, (Bk, pops), (C, cops))

    orig = rf.hashlib.blake2b
    if rebind_all:
        counter = [0]

        def salted(*a, **k):                 # every key unique -> everything rebinds
            counter[0] += 1
            h = orig(*a, **k); h.update(str(counter[0]).encode()); return h
        rf.hashlib.blake2b = salted
    try:
        res = fit_rows(rows_for, None, lo=LO, hi=HI, base_grid=(4, 4, 4, 4), n_stages=1,
                       steps=steps, cg_iters=8, resample_every=resample, ctx=ctx, verbose=False)
    finally:
        rf.hashlib.blake2b = orig
    return res.coef.copy(), list(res.stage_losses)


class TestRebind(unittest.TestCase):
    def test_diff_path_equals_rebind_all(self):
        ctx = context()
        c_diff, l_diff = run(ctx, False)
        c_all, l_all = run(ctx, True)
        r = np.abs(c_diff - c_all).max() / max(np.abs(c_all).max(), 1e-30)
        self.assertLess(r, 1e-4)
        self.assertLessEqual(abs(l_diff[-1] - l_all[-1]), 1e-5 * abs(l_all[-1]) + 1e-6)


if __name__ == "__main__":
    unittest.main()
