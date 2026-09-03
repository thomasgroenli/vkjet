"""The `s` column is redundant: a residual's ORDER-0 term can live in the
operator table like its order-1 and order-2 terms, with the per-row value riding
the payload.

    r = <L, J(f)> + J'QJ - s          (s column)
    r = <L, J(f)> + J'QJ - c[k]       (SLOT_CONST entry, payload slot k)

These must agree on loss, gradient, GN diagonal and HVP — not approximately: the
same arithmetic in a different place."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vkjet.context import Context                              # noqa: E402
from vkjet.data import Axes, rows_from_unified                 # noqa: E402
from vkjet.eqrow import (EqRowTerm, OperatorTable, SLOT_VAL,   # noqa: E402
                          SLOT_CONST, row_residuals_oracle)
from vkjet.apply import ApplyForward                           # noqa: E402


def _term(ctx, bases, iw, rows, ops, axes):
    t = EqRowTerm(ctx, bases, iw, ops)
    t.bind_batch(axes.encode(np.ascontiguousarray(rows["x"], np.float32)),
                 np.ascontiguousarray(rows["op"], np.int32),
                 np.ascontiguousarray(rows["w"], np.float32),
                 np.ascontiguousarray(rows["s"], np.float32),
                 np.ascontiguousarray(rows["c"], np.float32))
    return t


def main():
    ctx = Context()
    rng = np.random.default_rng(0)
    lo, hi = [0.0, 0.0, 0.0, 0.0], [1.0, 0.06, 0.06, 0.10]
    grid = (6, 6, 6, 8)
    axes = Axes(lo, hi, grid, periodic=(True, False, False, False))
    bases = axes.bases()
    ext = np.asarray(hi) - np.asarray(lo)
    iw = [grid[k] / ext[k] for k in range(4)]
    n = int(np.prod(grid)) * 5

    N = 5000
    x = np.stack([rng.uniform(l, h, N) for l, h in zip(lo, hi)], 1).astype(np.float32)
    d = rng.normal(size=(N, 5)).astype(np.float32)
    s = rng.normal(size=N).astype(np.float32)

    rows_s, ops_s = rows_from_unified(x, d, s)                    # s column
    rows_c, ops_c = rows_from_unified(x, d, s, target_cix=5)      # order-0 term
    assert any(e[0] == SLOT_CONST for e in ops_c.lin[0]), "no order-0 entry"
    assert not any(e[0] == SLOT_CONST for e in ops_s.lin[0])
    assert float(np.abs(rows_c["s"]).max()) == 0.0, "order-0 form must not use s"

    coef = rng.normal(size=n).astype(np.float32) * 0.1
    cbuf = ctx.buffer(n * 4); cbuf.upload(coef)
    vbuf = ctx.buffer(n * 4); vbuf.upload(rng.normal(size=n).astype(np.float32))

    out = {}
    for tag, (rws, ops) in (("s", (rows_s, ops_s)), ("const", (rows_c, ops_c))):
        t = _term(ctx, bases, iw, rws, ops, axes)
        lb = ctx.buffer(4); lb.zero(); t.loss(cbuf, lb)
        g = ctx.buffer(n * 4); g.zero(); t.accumulate(cbuf, g)
        dg = ctx.buffer(n * 4); dg.zero(); t.accumulate_diag(cbuf, dg)
        hv = ctx.buffer(n * 4); hv.zero(); t.hvp(cbuf, vbuf, hv)
        out[tag] = (float(lb.download(np.float32, 1)[0]),
                    g.download(np.float32, n), dg.download(np.float32, n),
                    hv.download(np.float32, n))
        del t

    names = ("loss", "grad", "diag", "hvp")
    for i, nm in enumerate(names):
        a, b = out["s"][i], out["const"][i]
        if i == 0:
            rel = abs(a - b) / max(abs(a), 1e-30)
        else:
            rel = float(np.abs(a - b).max() / (np.abs(a).max() + 1e-30))
        print(f"  {nm:5s}  s-form vs order-0 form: rel {rel:.3e}")
        assert rel < 1e-6, (nm, rel)

    # and both must match the numpy oracle
    af = ApplyForward(ctx, bases, 5)
    from vkjet.eqrow import gather_fields_jet
    xe = axes.encode(x)
    ext_t = tuple(b.primal_extent for b in bases)
    flds = gather_fields_jet(bases, iw, xe,
                             coef.astype(np.float64).reshape(ext_t + (5,)))
    r_s = row_residuals_oracle(flds, ops_s, rows_s["op"], rows_s["w"],
                               rows_s["s"], rows_s["c"])
    r_c = row_residuals_oracle(flds, ops_c, rows_c["op"], rows_c["w"],
                               rows_c["s"], rows_c["c"])
    rel = float(np.abs(r_s - r_c).max() / (np.abs(r_s).max() + 1e-30))
    print(f"  oracle residuals agree: rel {rel:.3e}")
    assert rel < 1e-9, rel
    L_or = float(0.5 * np.sum(rows_c["w"] * r_c * r_c))
    rel = abs(L_or - out["const"][0]) / max(abs(L_or), 1e-30)
    print(f"  GPU order-0 loss vs oracle: rel {rel:.3e}")
    assert rel < 1e-5, rel
    print("CONST TERM OK — the s column is redundant")
    ctx.destroy()




def test_jit_const():
    """The JIT must SPECIALIZE an order-0 operator, not fall back. In a JIT-only
    package (vkjet) the generic kernel is the referee, not a fast path, so the
    homogeneous form is unusable unless the emitter handles it."""
    from vkjet.genkernel import (GeneratedRowTerm, verify_generated,
                                  find_compiler, emit_shaders)
    comp = find_compiler()
    if comp is None:
        print("  no glslc/glslangValidator — skipping JIT arm")
        return
    ctx = Context()
    rng = np.random.default_rng(4)
    lo, hi = [0.0, 0.0, 0.0, 0.0], [1.0, 0.06, 0.06, 0.10]
    grid = (6, 6, 6, 8)
    axes = Axes(lo, hi, grid, periodic=(True, False, False, False))
    bases = axes.bases()
    ext = np.asarray(hi) - np.asarray(lo)
    iw = [grid[k] / ext[k] for k in range(4)]

    N = 3000
    x = np.stack([rng.uniform(l, h, N) for l, h in zip(lo, hi)], 1).astype(np.float32)
    d = rng.normal(size=(N, 5)).astype(np.float32)
    sv = rng.normal(size=N).astype(np.float32)
    rows_c, ops_c = rows_from_unified(x, d, sv, target_cix=5)

    # the emitted source must gather NO slot-15 site and carry the constant
    srcs = emit_shaders(ops_c, 0)[0] if isinstance(
        emit_shaders(ops_c, 0), tuple) else emit_shaders(ops_c, 0)
    grad_src = srcs["grad"] if isinstance(srcs, dict) else str(srcs)
    assert "f15_" not in grad_src, "emitted a gather for the order-0 sentinel slot"
    print("  emitted kernel gathers no slot-15 site: OK")

    gt = GeneratedRowTerm(ctx, bases, iw, ops_c, 0, compiler=comp)
    verify_generated(ctx, gt, bases, iw, ops_c, 0)     # parity vs the referee
    print("  verify_generated (JIT vs generic referee) PASSED for the order-0 op")
    del gt
    ctx.destroy()


if __name__ == "__main__":
    main()
    test_jit_const()
