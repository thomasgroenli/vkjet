"""SeparableTransfer == the assembled 4-D CSR (build_resize_csr), and Pᵀ is its
exact adjoint. The separable path is what makes BPX affordable at the production
grid (the assembled CSR is 2^nd nonzeros per FINE node PER LEVEL), so it has to
be the same operator, not merely a similar one."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vkjet.context import Context                             # noqa: E402
from vkjet.bpx import BpxPreconditioner                       # noqa: E402
from vkjet.rowfit import multilinear_resize, multilinear_restrict  # noqa: E402


def _levels(ctx, exts, nch, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for e in exts:
        n = int(np.prod(e)) * nch
        d = ctx.buffer(n * 4)
        d.upload(np.abs(rng.normal(1.0, 0.3, n)).astype(np.float32))
        out.append(dict(ext=tuple(e), diag=d, dmax=2.0))
    return out


def main():
    ctx = Context()
    nch = 5
    exts = [(4, 4, 4, 5), (8, 8, 8, 10), (16, 16, 16, 20)]
    ext_f = exts[-1]
    n_f = int(np.prod(ext_f)) * nch
    rng = np.random.default_rng(1)
    r = rng.normal(size=n_f).astype(np.float32)
    rb = ctx.buffer(n_f * 4); rb.upload(r)
    z1 = ctx.buffer(n_f * 4); z2 = ctx.buffer(n_f * 4)

    pa = BpxPreconditioner(ctx, ext_f, nch, _levels(ctx, exts, nch),
                           floor_rel=1e-2, transfer="csr")
    pb = BpxPreconditioner(ctx, ext_f, nch, _levels(ctx, exts, nch),
                           floor_rel=1e-2, transfer="separable")
    assert pa.transfer == "csr" and pb.transfer == "separable"
    pa.apply(rb, z1); pb.apply(rb, z2)
    a = z1.download(np.float32, n_f); b = z2.download(np.float32, n_f)
    rel = float(np.abs(a - b).max() / (np.abs(a).max() + 1e-30))
    print(f"  BPX apply  csr vs separable: rel {rel:.2e}")
    assert rel < 1e-5, rel

    # the transfers themselves, against the numpy reference pair
    L = [x for x in pb.levels if x["T"] is not None][0]
    ec = L["ext"]; nc = int(np.prod(ec)) * nch
    xc = rng.normal(size=nc).astype(np.float32)
    cb = ctx.buffer(nc * 4); cb.upload(xc)
    fb = ctx.buffer(n_f * 4)
    L["T"].prolong(cb, fb)
    ref = multilinear_resize(xc, ec, ext_f, nch)
    got = fb.download(np.float32, n_f)
    e1 = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30))
    L["T"].restrict(rb, cb)
    refr = multilinear_restrict(r, ext_f, ec, nch)
    gotr = cb.download(np.float32, nc)
    e2 = float(np.abs(gotr - refr).max() / (np.abs(refr).max() + 1e-30))
    print(f"  prolong vs multilinear_resize:   rel {e1:.2e}")
    print(f"  restrict vs multilinear_restrict: rel {e2:.2e}")
    assert e1 < 1e-5 and e2 < 1e-5, (e1, e2)

    # adjoint identity <Px, y> == <x, P'y>  (symmetry of C^-1)
    y = rng.normal(size=n_f).astype(np.float32)
    yb = ctx.buffer(n_f * 4); yb.upload(y)
    cb.upload(xc)                      # restrict above left P'r in cb
    L["T"].prolong(cb, fb); Px = fb.download(np.float32, n_f)
    L["T"].restrict(yb, cb); Pty = cb.download(np.float32, nc)
    lhs = float(np.dot(Px.astype(np.float64), y.astype(np.float64)))
    rhs = float(np.dot(xc.astype(np.float64), Pty.astype(np.float64)))
    print(f"  <Px,y> {lhs:.6f} vs <x,P'y> {rhs:.6f}")
    assert abs(lhs - rhs) <= 1e-4 * max(abs(lhs), 1.0), (lhs, rhs)

    # accumulate: prolong(accum=True) adds rather than overwrites
    cb.upload(xc)                      # restrict above left P'y in cb
    fb.upload(np.ones(n_f, np.float32))
    L["T"].prolong(cb, fb, accum=True)
    acc = fb.download(np.float32, n_f)
    e3 = float(np.abs(acc - (Px + 1.0)).max() / (np.abs(Px).max() + 1e-30))
    print(f"  prolong accum: rel {e3:.2e}")
    assert e3 < 1e-5, e3
    print("BPX SEPARABLE OK")
    ctx.destroy()




def test_level_weight():
    """omega_j = 2^(alpha*d_j) must scale each coarse level's contribution exactly,
    and alpha=0 must leave the flat sum untouched. Reference computed on the host
    from multilinear_resize/restrict."""
    ctx = Context()
    nch = 5
    exts = [(4, 4, 4, 5), (8, 8, 8, 10), (16, 16, 16, 20)]
    ext_f = exts[-1]
    n_f = int(np.prod(ext_f)) * nch
    rng = np.random.default_rng(3)
    r = rng.normal(size=n_f).astype(np.float32)
    rb = ctx.buffer(n_f * 4); rb.upload(r)
    zb = ctx.buffer(n_f * 4)
    floor_rel = 1e-2

    # fixed diagonals, reused across alphas so the arms are comparable
    diags = {e: np.abs(rng.normal(1.0, 0.3, int(np.prod(e)) * nch)) + 0.1
             for e in exts}

    def levels():
        out = []
        for e in exts:
            n = int(np.prod(e)) * nch
            d = ctx.buffer(n * 4); d.upload(diags[e].astype(np.float32))
            out.append(dict(ext=e, diag=d, dmax=float(diags[e].max())))
        return out

    for alpha in (0.0, 1.0, -1.0, 0.5):
        p = BpxPreconditioner(ctx, ext_f, nch, levels(), floor_rel=floor_rel,
                              transfer="separable", level_weight=alpha)
        p.apply(rb, zb)
        got = zb.download(np.float32, n_f).astype(np.float64)
        ref = np.zeros(n_f)
        for e in exts:
            d = diags[e]; fl = floor_rel * d.max()
            dj = max(np.log2(ext_f[k] / e[k]) for k in range(4))
            om = 2.0 ** (alpha * dj)
            if e == ext_f:
                ref += r / (d + fl)
            else:
                rc = multilinear_restrict(r, ext_f, e, nch)
                ref += om * multilinear_resize(rc / (d + fl), e, ext_f, nch)
        rel = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30))
        oms = [round(L["omega"], 4) for L in p.levels]
        print(f"  alpha={alpha:+.1f}  omega={oms}  rel {rel:.2e}")
        assert rel < 1e-4, (alpha, rel)
    print("LEVEL WEIGHT OK")
    ctx.destroy()


if __name__ == "__main__":
    main()
    test_level_weight()
