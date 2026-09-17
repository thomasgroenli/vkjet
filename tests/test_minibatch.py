"""Minibatched rows: the full solve must be the K=1 special case, not a
parallel path — and the buckets must be a partition of the one row system.

  1. K=1 is an ALIAS: the bucket is the full batch object, scale is exactly 1,
     every dispatch is the unwrapped one (bit-for-bit on the gradient, and
     end-to-end through fit_rows against minibatch=0).
  2. The buckets PARTITION the rows: forcing all K (tol=0) reproduces the
     full-batch gradient to atomic noise.
  3. A partial draw's error is consistent with the variance the norm test
     reports, and CG sees ONE operator (hvp replays the drawn set).
  4. The norm test passes once the whole row set is drawn (finite population).
  5. The JIT dispatch path binds every tier with the same partition.
Run: PYTHONPATH=~/projects/volkano python3 tests/test_minibatch.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vkjet import (Context, Axes, OperatorTable, make_rows, merge_row_sets, data_operator,   # noqa: E402
                   fit_rows, EqRowTerm, SLOT_VAL, SLOT_DX, SLOT_DY, SLOT_DZ)
from vkjet.rowfit import _jit_terms                                                          # noqa: E402
from vkjet.rowbatch import row_buckets, local_buckets, BatchedRowSystem                     # noqa: E402

rng = np.random.default_rng(11)
LO = np.array([0., 0., 0., 0.]); HI = np.array([1., .06, .06, .10]); GRID = (6, 6, 6, 10)
NC, ND = 4_000, 12_000


def build():
    """Two physics operators (one quadratic) at shared points + directional data rows."""
    ops = OperatorTable()
    cont = ops.add_op("continuity", lin=[(SLOT_DX, 0, 1.0), (SLOT_DY, 1, 1.0), (SLOT_DZ, 2, 1.0)])
    adv = ops.add_op("advection-x", lin=[(SLOT_DX, 3, 1.0)],
                     quad=[(SLOT_VAL, 0, SLOT_DX, 0, 1.0), (SLOT_VAL, 1, SLOT_DY, 0, 1.0)])
    ops, did = data_operator(ops)
    colloc = rng.uniform(LO, HI, (NC, 4)).astype(np.float32)
    eq = np.concatenate([make_rows(colloc, np.full(NC, k, np.int32), np.full(NC, 0.5, np.float32),
                                   np.zeros(NC, np.float32)) for k in (cont, adv)])
    x = rng.uniform(LO, HI, (ND, 4)).astype(np.float32)
    c = np.zeros((ND, 5), np.float32); c[:, :3] = rng.standard_normal((ND, 3))
    dr = make_rows(x, np.full(ND, did, np.int32), rng.uniform(0.2, 5.0, ND).astype(np.float32),
                   rng.standard_normal(ND).astype(np.float32), c)
    return np.concatenate([eq, dr]), ops


class H:
    def __init__(self, ctx, ops):
        self.ctx = ctx
        self.axes = Axes(LO, HI, GRID, periodic=(True, False, False, False)); self.bases = self.axes.bases()
        ext = HI - LO; self.iw = [GRID[k] / ext[k] for k in range(4)]
        e = tuple(b.primal_extent for b in self.bases); self.n = int(np.prod(e)) * 5
        self.cb = ctx.buffer(4 * self.n); self.cb.upload((rng.standard_normal(self.n) * 0.2).astype(np.float32))
        self.vb = ctx.buffer(4 * self.n); self.vb.upload(rng.standard_normal(self.n).astype(np.float32))
        self.g = ctx.buffer(4 * self.n); self.ops = ops

    def dl(self, buf): return buf.download(np.float32, self.n).astype(np.float64)

    def accum(self, obj):
        self.g.zero()
        for t in (obj if isinstance(obj, list) else [obj]): t.accumulate(self.cb, self.g)
        return self.dl(self.g)

    def hvp(self, obj):
        self.g.zero()
        for t in (obj if isinstance(obj, list) else [obj]): t.hvp(self.cb, self.vb, self.g)
        return self.dl(self.g)


def rel(a, b): return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main():
    ctx = Context()
    rows, ops = build()
    x, op, w, s, c = (np.ascontiguousarray(rows[k]) for k in ("x", "op", "w", "s", "c"))
    N = len(rows); h = H(ctx, ops); xe = h.axes.encode(x)
    print(f"{N:,} rows, {h.n:,} coefficients")
    for K in (1, 4, 7, 13):
        bk = row_buckets(N, K, seed=3); sz = sorted(len(b) for b in bk)
        assert len(bk) == K and np.array_equal(np.sort(np.concatenate(bk)), np.arange(N)) and sz[-1] - sz[0] <= 1
    assert np.array_equal(row_buckets(N, 1)[0], np.arange(N))
    print("0. row_buckets: partition exact, sizes differ by <=1, K=1 identity")

    ref = EqRowTerm(ctx, h.bases, h.iw, ops); ref.bind_batch(xe, op, w, s, c)
    g_full, hv_full = h.accum([ref]), h.hvp([ref])
    eta = max(rel(h.accum([ref]), g_full), rel(h.hvp([ref]), hv_full))
    t1 = EqRowTerm(ctx, h.bases, h.iw, ops); t1.bind_buckets(xe, op, w, s, c, row_buckets(N, 1))
    assert t1.minibatches[0] is t1._batch, "K=1 copied the batch"
    sys1 = BatchedRowSystem(ctx, [t1], h.n, tol=0.15)
    assert sys1.stochastic is False and sys1.K == 1
    g1, hv1 = h.accum(sys1), h.hvp(sys1); tok = sys1._batch; h.accum(sys1)
    print(f"1. K=1 alias: grad rel {rel(g1, g_full):.2e} hvp rel {rel(hv1, hv_full):.2e} (floor eta {eta:.2e}), "
          f"scale {sys1._scale!r}, capture token stable {sys1._batch is tok}")
    assert rel(g1, g_full) <= max(3 * eta, 1e-8) and rel(hv1, hv_full) <= max(3 * eta, 1e-8)
    assert sys1._scale == 1.0 and sys1._batch is tok

    for K in (4, 13):
        tk = EqRowTerm(ctx, h.bases, h.iw, ops); tk.bind_buckets(xe, op, w, s, c, row_buckets(N, K, seed=5))
        assert sum(b["n"] for b in tk.minibatches) == tk._batch["n"]
        sk = BatchedRowSystem(ctx, [tk], h.n, tol=0.0)
        gk, hvk = h.accum(sk), h.hvp(sk)
        assert sk.eff_batch == (K, K) and sk._scale == 1.0 and sk.grad_noise == 0.0
        print(f"2. K={K:<3} all buckets: grad rel {rel(gk, g_full):.2e} hvp rel {rel(hvk, hv_full):.2e}")
        assert rel(gk, g_full) < 2e-6 and rel(hvk, hv_full) < 2e-6

    K = 16
    tk = EqRowTerm(ctx, h.bases, h.iw, ops); tk.bind_buckets(xe, op, w, s, c, row_buckets(N, K, seed=9))
    sk = BatchedRowSystem(ctx, [tk], h.n, tol=0.5, seed=2)
    errs, ratios = [], []
    for trial in range(6):
        gk = h.accum(sk); used, sc = list(sk._used), sk._scale
        err = np.linalg.norm(gk - g_full); errs.append(err / np.linalg.norm(g_full))
        hvk = h.hvp(sk); man = np.zeros(h.n)
        for kk in used:
            h.g.zero(); tk.hvp(h.cb, h.vb, h.g, batch=tk.minibatches[kk]); man += h.dl(h.g)
        man *= sc
        assert rel(hvk, man) < 2e-6, (trial, rel(hvk, man))
        if len(used) == K:
            assert sk.grad_noise == 0.0 and errs[-1] < 1e-5; ratios.append(0.0)
        else:
            ratios.append(err / sk.grad_noise); assert 0.05 < ratios[-1] < 20.0, (trial, ratios[-1])
    print(f"3. K={K} tol=0.5: eff batch {sk.eff_batch[0]}/{K}, grad rel error {np.mean(errs):.3f}, "
          f"error/reported-sigma {np.mean(ratios):.2f}, hvp replays the drawn set")

    tk2 = EqRowTerm(ctx, h.bases, h.iw, ops); tk2.bind_buckets(xe, op, w, s, c, row_buckets(N, 8, seed=1))
    s2 = BatchedRowSystem(ctx, [tk2], h.n, tol=0.05, seed=4); h.accum(s2)
    print(f"4. strict tol: eff batch {s2.eff_batch[0]}/8, passed {s2.batcher.passed}, sigma {s2.grad_noise:.3g}")
    assert s2.batcher.passed and s2.eff_batch == (8, 8) and s2.grad_noise == 0.0

    of = np.empty(N, np.int32)
    for i, ix in enumerate(row_buckets(N, 8, seed=4)): of[ix] = i
    tt, handled = _jit_terms(ctx, h.axes, h.bases, GRID, x, op, w, s, c, ops, h.iw, False,
                             fuzz=np.zeros(N, np.float32), modulus=np.zeros(N, np.float32), of_row=of, nb=8)
    terms = [t for t, _ in tt]
    if not handled.all():
        gen = EqRowTerm(ctx, h.bases, h.iw, ops); r = np.flatnonzero(~handled)
        gen.bind_buckets(h.axes.encode(x[r]), op[r], w[r], s[r], c[r], local_buckets(of, r, 8)); terms.append(gen)
    sj = BatchedRowSystem(ctx, terms, h.n, tol=0.0); gj, hvj = h.accum(sj), h.hvp(sj)
    print(f"5. JIT dispatch, K=8 all buckets [{', '.join(sorted({type(t).__name__ for t in terms}))}]: "
          f"grad rel {rel(gj, g_full):.2e} hvp rel {rel(hvj, hv_full):.2e}")
    assert rel(gj, g_full) < 5e-5 and rel(hvj, hv_full) < 5e-5

    kw = dict(ops=ops, lo=LO, hi=HI, base_grid=GRID, n_stages=1, steps=6, cg_iters=6, ctx=ctx, seed=1, verbose=False)
    r0 = fit_rows(rows, **kw, minibatch=0); r1 = fit_rows(rows, **kw, minibatch=N)
    dl = abs(r0.stage_losses[-1] - r1.stage_losses[-1]) / abs(r0.stage_losses[-1])
    dc = float(np.max(np.abs(r0.coef - r1.coef))) / max(float(np.max(np.abs(r0.coef))), 1e-30)
    print(f"6. fit_rows minibatch=0 vs K=1: loss rel {dl:.2e}, max|dcoef| rel {dc:.2e}")
    assert dl < 1e-4 and dc < 1e-3, (dl, dc)
    r4 = fit_rows(rows, **kw, minibatch=N // 4, batch_tol=0.15)
    eff = r4.diagnostics["eff_batch"][0]
    print(f"7. fit_rows K=4: loss {r4.stage_losses[-1]:.5e} vs full {r0.stage_losses[-1]:.5e}, effective batch {np.mean(eff):.2f}/4")
    assert r4.stage_losses[-1] < r0.stage_losses[-1] * 1.5
    print("\nALL PASS"); ctx.destroy()


if __name__ == "__main__":
    main()
