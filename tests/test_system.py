"""The object layer is the same objective as fit_rows: a RowSystem of two
families built on a Field solves to the same coefficients as the concatenated
rows handed to fit_rows directly; the hash is order-independent; the world-unit
cell rounds up to the ladder.  Run: PYTHONPATH=~/projects/volkano python3 tests/test_system.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vkjet import (OperatorTable, make_rows, merge_row_sets, data_operator, fit_rows, Context,   # noqa: E402
                   rows_hash, SLOT_DX, SLOT_DY, SLOT_DZ)
from vkjet.system import Field, RowSystem, Solve                                                 # noqa: E402

LO, HI = (0., 0., 0., 0.), (1., 1., 1., 1.)


def truth(x):
    t, X, Y, Z = x.T
    return np.stack([np.sin(2*np.pi*Y) * np.cos(2*np.pi*t), np.sin(2*np.pi*Z), np.sin(2*np.pi*X), 0*X, 0*X], 1)


class Data:
    name = "data"; reference = True

    def build(self, system):
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 1, (5000, 4)).astype(np.float32); u = truth(x)
        d = rng.normal(size=(5000, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
        c = np.zeros((5000, 5), np.float32); c[:, :3] = d
        ops, did = data_operator(None)
        return make_rows(x, np.full(5000, did, np.int32), np.ones(5000, np.float32),
                         np.einsum("ij,ij->i", u[:, :3], d).astype(np.float32), c), ops


class Continuity:
    name = "continuity"; reference = False

    def __init__(self, mass): self.mass = mass

    def build(self, system):
        ops = OperatorTable()
        k = ops.add_op("continuity", lin=[(SLOT_DX, 0, 1.0), (SLOT_DY, 1, 1.0), (SLOT_DZ, 2, 1.0)])
        x = np.random.default_rng(1).uniform(0, 1, (3000, 4)).astype(np.float32)
        w = np.full(3000, self.mass * system.reference_weight / 3000, np.float32)
        return make_rows(x, np.full(3000, k, np.int32), w, np.zeros(3000, np.float32)), ops


def test_system():
    field = Field(LO, HI, n_channels=5, base_grid=(4, 4, 4, 4), n_stages=2)
    ctx = Context()
    sysA = RowSystem(field).add(Data()).add(Continuity(0.05))
    sysB = RowSystem(field).add(Continuity(0.05)).add(Data())
    assert sysA.hash == sysB.hash, "hash must not depend on family order"
    ra = sysA.fit(Solve(bpx=False, steps=6, cg_iters=8, verbose=False), ctx=ctx)
    rows, ops = merge_row_sets(*[sysA._built[f.name] for f in sysA.families])
    rb = fit_rows(rows, ops, lo=LO, hi=HI, base_grid=(4, 4, 4, 4), n_stages=2, steps=6, cg_iters=8,
                  ctx=ctx, verbose=False)
    rel = np.abs(ra.coef - rb.coef).max() / max(np.abs(rb.coef).max(), 1e-30)
    print(f"  system.fit vs fit_rows: max |dcoef| rel {rel:.2e}; hash {sysA.hash[:12]}")
    assert rel < 1e-5
    f2 = Field(LO, HI, base_grid=(4, 4, 4, 4), n_stages=3, cell=(None, 0.07, None, None))
    print(f"  world-unit cell 0.07 on unit box, 3 stages -> grid {f2.grid}, cells {[round(c, 4) for c in f2.cells]}")
    assert f2.grid[1] == 16 and f2.cells[1] <= 0.07
    ctx.destroy()
    print("SYSTEM OK")


if __name__ == "__main__":
    test_system()
