"""The channel count is DECLARED, not compiled in.

NCH used to be `#define NCH 5` in four shaders and a module constant in Python.
It is now a specialization constant carried by the OperatorTable (and so by the
row file), because the table references channel indices: the objective it
defines is not well posed without one. This checks that a table at a channel
count other than 5 gives exactly the numpy oracle."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vkjet.context import Context                               # noqa: E402
from vkjet.data import Axes, make_rows                          # noqa: E402
from vkjet.eqrow import (EqRowTerm, OperatorTable, SLOT_VAL,    # noqa: E402
                          SLOT_DT, SLOT_DX, SLOT_DZ, SLOT_DXX,
                          gather_fields_jet, row_residuals_oracle)


def check(ctx, nch, rng):
    lo, hi = [0.0, 0.0, 0.0, 0.0], [1.0, 0.06, 0.06, 0.10]
    grid = (6, 6, 6, 8)
    axes = Axes(lo, hi, grid, periodic=(True, False, False, False))
    bases = axes.bases()
    ext = np.asarray(hi) - np.asarray(lo)
    iw = [grid[k] / ext[k] for k in range(4)]
    n = int(np.prod(grid)) * nch

    ops = OperatorTable(n_channels=nch)
    top = nch - 1                       # exercise the HIGHEST channel index
    ops.add_op("mixed",
               lin=[(SLOT_VAL, 0, 1.0), (SLOT_DT, top, 0.7),
                    (SLOT_DXX, min(2, top), -0.3)],
               quad=[(SLOT_VAL, 0, SLOT_DX, top, 0.5)])
    ops.add_op("payload-covector",
               lin=[(SLOT_VAL, ch, 1.0, ch) for ch in range(min(nch, 8))],
               const=((-1.0, 7),) if nch <= 7 else ())

    N = 600
    x = np.stack([rng.uniform(l, h, N) for l, h in zip(lo, hi)], 1).astype(np.float32)
    op = rng.integers(0, ops.n_ops, N).astype(np.int32)
    w = np.abs(rng.normal(1.0, 0.2, N)).astype(np.float32)
    s = rng.normal(size=N).astype(np.float32)
    c = rng.normal(size=(N, 8)).astype(np.float32)
    rows = make_rows(x, op, w, s, c)

    t = EqRowTerm(ctx, bases, iw, ops)
    assert t.nch == nch, (t.nch, nch)
    t.bind_batch(axes.encode(x), op, w, s, c)
    coef = (rng.normal(size=n) * 0.1).astype(np.float32)
    cb = ctx.buffer(n * 4); cb.upload(coef)
    lb = ctx.buffer(4); lb.zero(); t.loss(cb, lb)
    g = ctx.buffer(n * 4); g.zero(); t.accumulate(cb, g)
    L_gpu = float(lb.download(np.float32, 1)[0])
    g_gpu = g.download(np.float32, n)

    extents = tuple(b.primal_extent for b in bases)
    flds = gather_fields_jet(bases, iw, axes.encode(x),
                             coef.astype(np.float64).reshape(extents + (nch,)))
    r = row_residuals_oracle(flds, ops, op, w, s, c)
    L_or = float(0.5 * np.sum(w.astype(np.float64) * r * r))
    rel = abs(L_gpu - L_or) / max(abs(L_or), 1e-30)
    # gradient by central differences on a few random coordinates
    probe = rng.choice(n, 2, replace=False)
    gerr = 0.0
    for i in probe:
        eps = 1e-3
        dl = []
        for sgn in (1.0, -1.0):
            cc = coef.astype(np.float64).copy(); cc[i] += sgn * eps
            f2 = gather_fields_jet(bases, iw, axes.encode(x),
                                   cc.reshape(extents + (nch,)))
            r2 = row_residuals_oracle(f2, ops, op, w, s, c)
            dl.append(0.5 * np.sum(w.astype(np.float64) * r2 * r2))
        fd = (dl[0] - dl[1]) / (2 * eps)
        gerr = max(gerr, abs(fd - g_gpu[i]) / (abs(fd) + 1.0))
    print(f"  NCH={nch}: loss rel {rel:.2e}   grad vs FD (2 probes) {gerr:.2e}")
    assert rel < 1e-5 and gerr < 1e-3, (nch, rel, gerr)
    del t


def main():
    ctx = Context()
    rng = np.random.default_rng(0)
    for nch in (3, 7):        # NCH=5 is covered exhaustively by test_eqrow
        check(ctx, nch, rng)
    print("N-CHANNEL OK — the channel count is data, not a compile-time constant")
    ctx.destroy()


if __name__ == "__main__":
    main()
