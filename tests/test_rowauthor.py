"""Authoring-side folding: (x, op, w, s, c, m) must define the SAME objective
as the (x, op, payload, m) the solver is handed.

With every coefficient on a payload slot the residual is homogeneous of degree
1 in the payload, so w·r(c)² = r(√w·c)²; a wrapped row's loss is homogeneous of
degree 2 in (r, m), so w·L(r; m) = L(√w·r; √w·m). This checks both on a
production-shaped operator set — momentum-like rows with literal coefficients
and quadratic advection, continuity, a per-row-payload (frame) continuity, and
data rows carrying their target in `s` (which must migrate to SLOT_CONST
before it can fold), a majority of them wrapped with per-row moduli.

  1. homogenize + fold reproduces the f64 oracle loss and the GPU
     loss/grad/diag/hvp, on the generic referee and the JIT tier, at two
     temperatures (wrapped Gaussian and the pure cosine)
  2. reweight AFTER folding == changing w BEFORE folding (the folded form is
     not write-hostile)
  3. the refusals fire: s with no target slot, non-homogeneous operator
  4. gain_of recovers w where a gain slot exists, NaN where it honestly cannot
  5. the folded rows hash differently from the unfolded ones (they are a
     different file) but are exactly one objective: rows_hash of two folds of
     the same system agree

Run: .venv/bin/python tests/test_rowauthor.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vkjet.context import Context                                   # noqa: E402
from vkjet.data import Axes, make_rows, data_operator, rows_hash    # noqa: E402
from vkjet.eqrow import (EqRowTerm, OperatorTable, row_loss_oracle,  # noqa: E402
                          SLOT_VAL, SLOT_DT, SLOT_DX, SLOT_DY, SLOT_DZ,
                          SLOT_DXX, SLOT_DYY, SLOT_DZZ)
from vkjet import genkernel as gk                                   # noqa: E402
from vkjet.rowauthor import (homogenize, fold, reweight,            # noqa: E402
                              gain_of, is_homogeneous)

rng = np.random.default_rng(7)
LO = np.array([0., 0., 0., 0.])
HI = np.array([1., .06, .06, .10])
GRID = (4, 4, 4, 6)
NC, ND = 300, 600          # the f64 oracle is a per-row Python gather; the
                           # identity is per-row exact, so this only has to
                           # exercise every operator class


def physics_ops(Lh=0.1, nu=1e-3):
    """Literal-coefficient operators (need a gain slot) and one whose
    coefficients already ride the payload (folds as is)."""
    ops = OperatorTable()
    adv = lambda ch: [(SLOT_VAL, 0, SLOT_DX, ch, Lh), (SLOT_VAL, 1, SLOT_DY, ch, Lh),
                      (SLOT_VAL, 2, SLOT_DZ, ch, Lh)]
    lap = lambda ch: [(SLOT_DXX, ch, -nu * Lh * Lh), (SLOT_DYY, ch, -nu * Lh * Lh),
                      (SLOT_DZZ, ch, -nu * Lh * Lh)]
    grad = [SLOT_DX, SLOT_DY, SLOT_DZ]
    ids = {}
    for ch, nm in enumerate("uvw"):
        ids[f"mom-{nm}"] = ops.add_op(f"mom-{nm}",
                                      lin=[(SLOT_DT, ch, Lh), (grad[ch], 3, Lh)] + lap(ch),
                                      quad=adv(ch))
    ids["cont"] = ops.add_op("cont", lin=[(SLOT_DX, 0, Lh), (SLOT_DY, 1, Lh), (SLOT_DZ, 2, Lh)])
    # frame continuity: five payload coefficients, homogeneous already
    ids["cont-frame"] = ops.add_op("cont-frame",
                                   lin=[(SLOT_DX, 0, Lh, 0), (SLOT_DZ, 0, Lh, 1),
                                        (SLOT_VAL, 0, Lh, 2), (SLOT_DX, 1, Lh, 3),
                                        (SLOT_DZ, 1, Lh, 4)])
    return ops, ids


def build():
    ops, ids = physics_ops()
    ops, did = data_operator(ops)
    sets = []
    for nm, oid in ids.items():
        x = rng.uniform(LO, HI, (NC, 4)).astype(np.float32)
        pay = None
        if nm == "cont-frame":
            a = rng.uniform(-0.5, 0.5, NC); rr = rng.uniform(0.02, 0.1, NC)
            pay = np.stack([np.sin(a), np.cos(a), 1 / rr, np.cos(a), -np.sin(a)], 1)
        w = rng.uniform(0.2, 5.0, NC).astype(np.float32)
        sets.append(make_rows(x, np.full(NC, oid, np.int32), w, np.zeros(NC, np.float32),
                              pay, fuzz=0.0))
    x = rng.uniform(LO, HI, (ND, 4)).astype(np.float32)
    c = np.zeros((ND, 5), np.float32); c[:, :3] = rng.standard_normal((ND, 3))
    mod = np.where(rng.random(ND) < 0.7, 0.5, 0.0).astype(np.float32)   # m = 2 venc
    dr = make_rows(x, np.full(ND, did, np.int32),
                   rng.uniform(0.2, 5.0, ND).astype(np.float32),
                   rng.uniform(-1.5, 1.5, ND).astype(np.float32), c, nyquist=mod)
    sets.append(dr)
    return np.concatenate(sets), ops, did


class H:
    def __init__(self, ctx):
        self.axes = Axes(LO, HI, GRID, periodic=(True, False, False, False))
        self.bases = self.axes.bases()
        ext = np.asarray(HI) - np.asarray(LO)
        self.iw = [GRID[k] / ext[k] for k in range(4)]
        e = tuple(b.primal_extent for b in self.bases)
        self.n = int(np.prod(e)) * 5
        self.C = rng.standard_normal(e + (5,)) * 0.4
        self.ctx = ctx
        self.cb = ctx.buffer(4 * self.n)
        self.cb.upload(np.ascontiguousarray(self.C.reshape(-1), np.float32))
        self.vb = ctx.buffer(4 * self.n)
        self.vb.upload(rng.standard_normal(self.n).astype(np.float32))
        self.o = [ctx.buffer(4 * self.n) for _ in range(3)]
        self.lb = ctx.buffer(4)

    def oracle(self, r, ops, tau):
        return row_loss_oracle(self.bases, self.iw, self.axes.encode(r["x"]),
                               self.C, ops, r["op"],
                               r["w"].astype(np.float64),
                               r["s"].astype(np.float64),
                               r["c"].astype(np.float64), 1.0,
                               modulus=r["nyquist"].astype(np.float64), tau=tau)

    def gpu(self, r, ops, tau, tier):
        """Loss/grad/diag/hvp of the whole row set on one tier. The JIT tier is
        one term per operator (rows bound against that operator's own
        sub-table), all accumulating into the same buffers."""
        for b in self.o:
            b.zero()
        self.lb.zero()
        xe = self.axes.encode(r["x"])
        if tier == "generic":
            parts = [(EqRowTerm(self.ctx, self.bases, self.iw, ops), slice(None), r["op"])]
        else:
            parts = []
            for k in np.unique(r["op"]):
                m = r["op"] == k
                parts.append((gk.GeneratedRowTerm(self.ctx, self.bases, self.iw, ops, int(k)),
                              m, np.zeros(int(m.sum()), np.int32)))
        for t, m, op in parts:
            t.tau = tau
            # FULL payload: homogenize puts the data target on a high slot, and a
            # [:, :5] truncation drops it in a way ONLY the gradient reveals
            # (loss and curvature survive: a constant does not enter grad r).
            t.bind_batch(xe[m], op, r["w"][m], r["s"][m], r["c"][m],
                         modulus=r["nyquist"][m])
            t.loss(self.cb, self.lb)
            t.accumulate(self.cb, self.o[0])
            t.accumulate_diag(self.cb, self.o[1])
            t.hvp(self.cb, self.vb, self.o[2])
        out = [float(self.lb.download(np.float32, 1)[0])]
        out += [b.download(np.float32, self.n).astype(np.float64) for b in self.o]
        del parts
        return out


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - b)
                 / max(np.linalg.norm(np.asarray(b)), 1e-30))


def main():
    ctx = Context()
    print(f"device: {ctx.device_name}")
    h = H(ctx)
    rows, ops, did = build()
    ops2, plan = homogenize(ops, absorb_s=["data"])
    folded = fold(rows, ops2, plan)

    print("operator table after homogenize:")
    for k in range(ops.n_ops):
        print(f"  {ops.names[k]:>12}: homog {str(is_homogeneous(ops, k)):>5} -> "
              f"{str(is_homogeneous(ops2, k)):>5}  gain {plan[k]['gain']}  "
              f"target {plan[k]['target']}  hint "
              f"{ops.kernels[k] or '(none)'!r} -> {ops2.kernels[k] or '(none)'!r}")
    wr = rows["nyquist"] > 0
    print(f"rows: w range [{rows['w'].min():.3g}, {rows['w'].max():.3g}], "
          f"|s| max {np.abs(rows['s']).max():.3g}, {int(wr.sum())} wrapped (m 0.5)")
    print(f"folded: w range [{folded['w'].min():g}, {folded['w'].max():g}], "
          f"|s| max {np.abs(folded['s']).max():g}, "
          f"m range [{folded['nyquist'][wr].min():.3g}, {folded['nyquist'][wr].max():.3g}]")

    # 1. same objective — oracle and both GPU tiers, wrapped Gaussian and cosine
    tiers = ["generic"] + (["jit"] if gk.find_compiler() else [])
    for tau in (0.25, -1.0):
        a, b = h.oracle(rows, ops, tau), h.oracle(folded, ops2, tau)
        print(f"\n1. tau {tau:+.2f}  f64 oracle {a:.12e} / folded {b:.12e}   "
              f"rel {abs(a - b) / abs(a):.2e}")
        assert abs(a - b) / abs(a) < 1e-6, "fold changed the objective"
        for tier in tiers:
            g0 = h.gpu(rows, ops, tau, tier); g1 = h.gpu(folded, ops2, tau, tier)
            errs = [abs(g1[0] - g0[0]) / abs(g0[0])] + [rel(g1[i], g0[i]) for i in (1, 2, 3)]
            print(f"   {tier:8s} loss {errs[0]:.2e} (vs oracle {abs(g0[0]-a)/abs(a):.2e})  "
                  + "  ".join(f"{nm} {e:.2e}" for nm, e in zip(("grad", "diag", "hvp"), errs[1:])))
            assert abs(g0[0] - a) / abs(a) < 1e-4
            assert errs[0] < 1e-5 and all(e < 1e-4 for e in errs[1:]), (tier, tau, errs)

    # 2. reweight after folding == reweighting before (wrapped rows included)
    k = rng.uniform(0.3, 4.0, len(rows))
    pre = rows.copy(); pre["w"] = pre["w"] * k.astype(np.float32)
    post = reweight(folded, k)
    a2, b2 = h.oracle(pre, ops, 0.25), h.oracle(post, ops2, 0.25)
    print(f"\n2. reweight before {a2:.10e} / after {b2:.10e}   rel "
          f"{abs(a2 - b2) / abs(a2):.2e}")
    assert abs(a2 - b2) / abs(a2) < 1e-6, "folded form is write-hostile"
    sub = rows["op"] == did
    post2 = reweight(folded, 2.0, mask=sub)
    pre2 = rows.copy(); pre2["w"][sub] *= 2
    assert abs(h.oracle(pre2, ops, 0.25) - h.oracle(post2, ops2, 0.25)) < 1e-6 * a2

    # 3. the refusals fire
    for why, fn in (
            ("s with no target slot",
             lambda: fold(rows, *homogenize(ops))),
            ("non-homogeneous operator",
             lambda: fold(rows, ops, {k: {"gain": None, "target": None}
                                      for k in range(ops.n_ops)}))):
        try:
            fn()
        except ValueError:
            print(f"3. refused: {why}")
        else:
            raise AssertionError(f"{why} was NOT refused")

    # 4. gain recovery, honest about where it is impossible
    w = gain_of(folded, ops2, plan)
    phys = ~np.isin(rows["op"], [did, ops.names.index("cont-frame")])
    err = np.abs(w[phys] - rows["w"][phys]) / rows["w"][phys]
    print(f"4. gain_of: literal-coefficient physics w recovered to {err.max():.2e}; "
          f"data and frame rows NaN (inseparable): {np.isnan(w[~phys]).all()}")
    assert err.max() < 1e-6 and np.isnan(w[~phys]).all()

    # 5. one objective, one hash; a different file from the unfolded rows
    assert rows_hash(folded, ops2) == rows_hash(fold(rows, ops2, plan), ops2)
    assert rows_hash(folded, ops2) != rows_hash(rows, ops)
    print("5. folded hash deterministic and distinct from the unfolded file")

    print("\nALL PASS")
    ctx.destroy()


if __name__ == "__main__":
    main()
