"""Stopping at stabilisation: with stop_rel > 0 the solve stops before the
budget once mean(pred/L) over the window is below stop_rel, the measured floor
is below stop_rel, and running on to the full budget changes the loss by no
more than the window-sum of what the rule allowed. It is a convergence
statement only.  Run: PYTHONPATH=~/projects/volkano python3 tests/test_stop.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vkjet import (OperatorTable, make_rows, merge_row_sets, data_operator,   # noqa: E402
                   fit_rows, Context, SLOT_DX, SLOT_DY, SLOT_DZ)

LO, HI = (0., 0., 0., 0.), (1., 1., 1., 1.)


def truth(x):
    t, X, Y, Z = x.T
    return np.stack([np.sin(2*np.pi*Y) * np.cos(2*np.pi*t), np.sin(2*np.pi*Z), np.sin(2*np.pi*X),
                     0*X, 0*X], 1)


def system(fuzz):
    rng = np.random.default_rng(0)
    xa = rng.uniform(0, 1, (6000, 4)).astype(np.float32)
    ua = truth(xa); d = rng.normal(size=(6000, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    c = np.zeros((6000, 5), np.float32); c[:, :3] = d
    ops, did = data_operator(None)
    A = make_rows(xa, np.full(6000, did, np.int32), np.ones(6000, np.float32),
                  (np.einsum("ij,ij->i", ua[:, :3], d) + 0.05 * rng.normal(size=6000)).astype(np.float32), c)
    pops = OperatorTable()
    cid = pops.add_op("continuity", lin=[(SLOT_DX, 0, 1.0), (SLOT_DY, 1, 1.0), (SLOT_DZ, 2, 1.0)])
    xb = rng.uniform(0, 1, (4000, 4)).astype(np.float32)
    B = make_rows(xb, np.full(4000, cid, np.int32), np.full(4000, 0.05, np.float32),
                  np.zeros(4000, np.float32), fuzz=fuzz)
    return merge_row_sets((A, ops), (B, pops))


def run(fuzz, stop_rel, steps=120):
    ctx = Context()
    rows, ops = system(fuzz)
    res = fit_rows(rows, ops, lo=LO, hi=HI, base_grid=(4, 4, 4, 4), n_stages=1, steps=steps,
                   cg_iters=8, stop_rel=stop_rel, stop_window=5, ctx=ctx, verbose=False)
    out = dict(loss=res.stage_losses[-1], used=res.diagnostics["steps_used"],
               floor=res.diagnostics["floor"], trace=res.diagnostics["pred_rel"])
    ctx.destroy()
    return out


def test_stop():
    # deterministic objective: the floor is arithmetic, the rule fires before the budget
    a = run(0.0, stop_rel=1e-4)
    b = run(0.0, stop_rel=0.0)
    print(f"  no jitter: floor {a['floor']:.2e}; stopped at {a['used']}/120, loss {a['loss']:.6g}; "
          f"full budget loss {b['loss']:.6g} (rel diff {abs(a['loss']-b['loss'])/b['loss']:.2e})")
    assert a["floor"] < 1e-4, "arithmetic floor above the threshold"
    assert a["used"] < 120, "did not stop before the budget"
    # what the rule allowed to remain: at most ~stop_rel per step over the rest of the budget
    assert abs(a["loss"] - b["loss"]) / b["loss"] < 1e-4 * (120 - a["used"]) + 1e-3
    # stochastic objective (per-step jitter): the statistical floor is the draw-to-draw
    # gradient spread, far above 1e-4, and the rule must NOT declare stabilisation
    c = run(0.5, stop_rel=1e-4)
    print(f"  jitter 0.5: floor {c['floor']:.2e} (statistical); used {c['used']}/120")
    assert c["floor"] > 1e-4, "jitter floor should exceed the threshold"
    assert c["used"] == 120, "must not stop on a stochastic objective below its floor"
    print("STOP OK — fires on a deterministic objective, refuses below the measured floor")


if __name__ == "__main__":
    test_stop()
