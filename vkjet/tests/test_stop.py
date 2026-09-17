"""Stopping at stabilisation: with stop_rel > 0 the solve stops before the
budget once mean(pred/L) over the window is below stop_rel and the measured
floor is below stop_rel; running on to the full budget then changes the loss
by no more than the window-sum of what the rule allowed. On a stochastic
objective (jittered rows) the statistical floor is far above the threshold
and the rule must NOT declare stabilisation. A convergence statement only."""
import unittest

import numpy as np

from vkjet import (OperatorTable, make_rows, merge_row_sets, data_operator, fit_rows,
                   SLOT_DX, SLOT_DY, SLOT_DZ)
from vkjet.tests import context

LO, HI = (0., 0., 0., 0.), (1., 1., 1., 1.)
STEPS = 80


def truth(x):
    t, X, Y, Z = x.T
    return np.stack([np.sin(2*np.pi*Y) * np.cos(2*np.pi*t), np.sin(2*np.pi*Z), np.sin(2*np.pi*X),
                     0*X, 0*X], 1)


def system(fuzz):
    rng = np.random.default_rng(0)
    n = 4000
    xa = rng.uniform(0, 1, (n, 4)).astype(np.float32)
    ua = truth(xa); d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    c = np.zeros((n, 5), np.float32); c[:, :3] = d
    ops, did = data_operator(None)
    A = make_rows(xa, np.full(n, did, np.int32), np.ones(n, np.float32),
                  (np.einsum("ij,ij->i", ua[:, :3], d) + 0.05 * rng.normal(size=n)).astype(np.float32), c)
    pops = OperatorTable()
    cid = pops.add_op("continuity", lin=[(SLOT_DX, 0, 1.0), (SLOT_DY, 1, 1.0), (SLOT_DZ, 2, 1.0)])
    xb = rng.uniform(0, 1, (3000, 4)).astype(np.float32)
    B = make_rows(xb, np.full(3000, cid, np.int32), np.full(3000, 0.05, np.float32),
                  np.zeros(3000, np.float32), fuzz=fuzz)
    return merge_row_sets((A, ops), (B, pops))


def run(ctx, fuzz, stop_rel):
    rows, ops = system(fuzz)
    res = fit_rows(rows, ops, lo=LO, hi=HI, base_grid=(4, 4, 4, 4), n_stages=1, steps=STEPS,
                   cg_iters=8, stop_rel=stop_rel, stop_window=5, ctx=ctx, verbose=False)
    return dict(loss=res.stage_losses[-1], used=res.diagnostics["steps_used"],
                floor=res.diagnostics["floor"])


class TestStop(unittest.TestCase):
    def test_deterministic_objective_stops_early(self):
        ctx = context()
        a = run(ctx, 0.0, stop_rel=1e-4)
        b = run(ctx, 0.0, stop_rel=0.0)
        self.assertLess(a["floor"], 1e-4, "arithmetic floor above the threshold")
        self.assertLess(a["used"], STEPS, "did not stop before the budget")
        self.assertLess(abs(a["loss"] - b["loss"]) / b["loss"], 1e-4 * (STEPS - a["used"]) + 1e-3)

    def test_stochastic_objective_refuses(self):
        c = run(context(), 0.5, stop_rel=1e-4)
        self.assertGreater(c["floor"], 1e-4, "jitter floor should exceed the threshold")
        self.assertEqual(c["used"], STEPS, "must not stop on a stochastic objective below its floor")


if __name__ == "__main__":
    unittest.main()
