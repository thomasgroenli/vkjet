"""The object layer is the same objective as fit_rows: a RowSystem of two
families built on a Field solves to the same coefficients as the concatenated
rows handed to fit_rows directly; the hash is order-independent; the
world-unit cell rounds up to the ladder."""
import unittest

import numpy as np

from vkjet import (OperatorTable, make_rows, merge_row_sets, data_operator, fit_rows,
                   SLOT_DX, SLOT_DY, SLOT_DZ)
from vkjet.system import Field, RowSystem, Solve
from vkjet.tests import context

LO, HI = (0., 0., 0., 0.), (1., 1., 1., 1.)


def truth(x):
    t, X, Y, Z = x.T
    return np.stack([np.sin(2*np.pi*Y) * np.cos(2*np.pi*t), np.sin(2*np.pi*Z), np.sin(2*np.pi*X),
                     0*X, 0*X], 1)


class Data:
    name = "data"; reference = True

    def build(self, system):
        rng = np.random.default_rng(0)
        n = 4000
        x = rng.uniform(0, 1, (n, 4)).astype(np.float32); u = truth(x)
        d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
        c = np.zeros((n, 5), np.float32); c[:, :3] = d
        ops, did = data_operator(None)
        return make_rows(x, np.full(n, did, np.int32), np.ones(n, np.float32),
                         np.einsum("ij,ij->i", u[:, :3], d).astype(np.float32), c), ops


class Continuity:
    name = "continuity"; reference = False

    def __init__(self, mass): self.mass = mass

    def build(self, system):
        ops = OperatorTable()
        k = ops.add_op("continuity", lin=[(SLOT_DX, 0, 1.0), (SLOT_DY, 1, 1.0), (SLOT_DZ, 2, 1.0)])
        n = 2000
        x = np.random.default_rng(1).uniform(0, 1, (n, 4)).astype(np.float32)
        w = np.full(n, self.mass * system.reference_weight / n, np.float32)
        return make_rows(x, np.full(n, k, np.int32), w, np.zeros(n, np.float32)), ops


class TestSystem(unittest.TestCase):
    def test_hash_is_order_independent(self):
        field = Field(LO, HI, n_channels=5, base_grid=(4, 4, 4, 4), n_stages=2)
        a = RowSystem(field).add(Data()).add(Continuity(0.05))
        b = RowSystem(field).add(Continuity(0.05)).add(Data())
        self.assertEqual(a.hash, b.hash)

    def test_system_fit_equals_fit_rows(self):
        ctx = context()
        field = Field(LO, HI, n_channels=5, base_grid=(4, 4, 4, 4), n_stages=2)
        sysA = RowSystem(field).add(Data()).add(Continuity(0.05))
        ra = sysA.fit(Solve(bpx=False, steps=6, cg_iters=8, verbose=False), ctx=ctx)
        rows, ops = merge_row_sets(*[sysA._built[f.name] for f in sysA.families])
        rb = fit_rows(rows, ops, lo=LO, hi=HI, base_grid=(4, 4, 4, 4), n_stages=2, steps=6,
                      cg_iters=8, ctx=ctx, verbose=False)
        r = np.abs(ra.coef - rb.coef).max() / max(np.abs(rb.coef).max(), 1e-30)
        self.assertLess(r, 1e-5)

    def test_world_unit_cell_rounds_up_the_ladder(self):
        f2 = Field(LO, HI, base_grid=(4, 4, 4, 4), n_stages=3, cell=(None, 0.07, None, None))
        self.assertEqual(f2.grid[1], 16)
        self.assertLessEqual(f2.cells[1], 0.07)


if __name__ == "__main__":
    unittest.main()
