"""Shared fixtures: a small 4D grid, a domain-free operator set spanning every
shape the kernels must handle, and random rows over it."""
from __future__ import annotations

import numpy as np

from vkjet.data import Axes, make_rows
from vkjet.eqrow import (OperatorTable, NCH, NCPR, SLOT_VAL, SLOT_DT, SLOT_DX,
                         SLOT_DY, SLOT_DZ, SLOT_DXX, SLOT_DYY, SLOT_DZZ,
                         SLOT_DTX, SLOT_DYZ)

LO, HI = (0., 0., 0., 0.), (1., 6., 6., 10.)
GRID = (4, 4, 4, 6)


def setup(grid=GRID, lo=LO, hi=HI, nch=NCH):
    """→ (axes, bases, inv_widths, n_coefficients) on a periodic-in-t grid."""
    axes = Axes(lo, hi, grid, periodic=(True, False, False, False))
    bases = axes.bases()
    iw = [grid[k] / (hi[k] - lo[k]) for k in range(4)]
    nco = int(np.prod([b.primal_extent for b in bases])) * nch
    return axes, bases, iw, nco


def sample_operators():
    """Linear-only, quadratic, mixed-derivative and payload operators. Stands
    in for any application's physics — vkjet itself is domain-free."""
    ops, ids = OperatorTable(), {}
    sp = (SLOT_DX, SLOT_DY, SLOT_DZ)
    for i in range(3):                      # advection-shaped: lin + quad
        lin = [(SLOT_DT, i, 1.0), (sp[i], 3, 1.0)]
        lin += [(dd, i, -1e-3) for dd in (SLOT_DXX, SLOT_DYY, SLOT_DZZ)]
        quad = [(SLOT_VAL, k, sp[k], i, 1.0) for k in range(3)]
        quad += [(SLOT_VAL, i, sp[k], k, 0.5) for k in range(3)]
        ids[f"mom{i}"] = ops.add_op(f"mom-{i}", lin, quad)
    ids["cont"] = ops.add_op("continuity", lin=[(SLOT_DX, 0, 1.0), (SLOT_DY, 1, 1.0),
                                                (SLOT_DZ, 2, 1.0)])
    ids["of"] = ops.add_op("transport", lin=[(SLOT_DT, 4, 1.0)],
                           quad=[(SLOT_VAL, k, sp[k], 4, 1.0) for k in range(3)])
    ids["data"] = ops.add_op("data", lin=[(SLOT_VAL, c, 1.0, c) for c in range(5)])
    ids["mixed"] = ops.add_op(
        "mixed", lin=[(SLOT_DXX, 1, .7), (SLOT_DTX, 0, -.3), (SLOT_DYZ, 4, .2)],
        quad=[(SLOT_DY, 2, SLOT_DY, 2, .5), (SLOT_VAL, 3, SLOT_DZ, 3, -.4)])
    return ops, ids


def random_rows(rng, ops, n, x_enc=None, extents=None):
    """n random rows over `ops` (encoded points, weights, targets, payload)."""
    if x_enc is None:
        g = np.asarray(extents, np.float64)
        x_enc = (rng.uniform(0, 1, (n, 4)) * g).astype(np.float32)
    op = rng.integers(0, ops.n_ops, n).astype(np.int32)
    w = rng.uniform(0.2, 2.0, n).astype(np.float32)
    s = (rng.standard_normal(n) * 0.2).astype(np.float32)
    c = rng.standard_normal((n, NCPR)).astype(np.float32)
    return x_enc, op, w, s, c


def four_way(ctx, ref, term, nco, seed=1):
    """Relative errors of `term` against `ref` on loss / grad / diag / hvp at a
    random coefficient vector (both terms already bound)."""
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
    out = {"loss": abs(b[0] - a[0]) / max(abs(a[0]), 1e-9)}
    for name, x0, x1 in zip(("grad", "diag", "hvp"), a[1:], b[1:]):
        out[name] = float(np.linalg.norm(x1 - x0) / max(np.linalg.norm(x0), 1e-9))
    return out


def divfree_truth(x):
    """An ABC-type field on the unit box: exactly divergence-free, fully 3D,
    time-modulated; each component depends only on the two coordinates it is
    not differentiated by."""
    tt, xx, yy, zz = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    g = 1.0 + 0.3 * np.cos(2 * np.pi * tt)
    u = (np.sin(np.pi * zz) + np.cos(np.pi * yy)) * g
    v = (np.sin(np.pi * xx) + np.cos(np.pi * zz)) * g
    w = (np.sin(np.pi * yy) + np.cos(np.pi * xx)) * g
    return np.stack([u, v, w, np.zeros_like(u), np.zeros_like(u)], 1)


def directional_rows(rng, x, U, did, noise=0.0, w=1.0):
    """Data rows r = <dir, u> - s with random unit directions in the velocity
    channels (payload = the covector)."""
    n = len(x)
    d = rng.standard_normal((n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    c = np.zeros((n, 5), np.float32); c[:, :3] = d
    s = (U[:, :3] * d).sum(1) + noise * rng.standard_normal(n)
    return make_rows(x, np.full(n, did, np.int32), np.full(n, w, np.float32),
                     s.astype(np.float32), c)
