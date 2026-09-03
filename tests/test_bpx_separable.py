"""The separable transfer must BE the tensor-product operator, and Pᵀ must be
its exact adjoint — otherwise C⁻¹ is not symmetric and CG is not valid."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vkjet.context import Context                                   # noqa: E402
from vkjet.bpx import BpxPreconditioner, FLOOR_REL                  # noqa: E402
from vkjet.rowfit import multilinear_resize, multilinear_restrict   # noqa: E402


def main():
    ctx = Context()
    nch = 5
    exts = [(4, 4, 4, 5), (8, 8, 8, 10), (16, 16, 16, 20)]
    ext_f = exts[-1]
    n_f = int(np.prod(ext_f)) * nch
    rng = np.random.default_rng(1)
    r = rng.normal(size=n_f).astype(np.float32)
    rb = ctx.buffer(n_f * 4); rb.upload(r)

    levels = []
    diags = {}
    for e in exts:
        n = int(np.prod(e)) * nch
        d = np.abs(rng.normal(1.0, 0.3, n)) + 0.1
        diags[e] = d
        buf = ctx.buffer(n * 4); buf.upload(d.astype(np.float32))
        levels.append(dict(ext=e, diag=buf, dmax=float(d.max())))
    pre = BpxPreconditioner(ctx, ext_f, nch, levels)

    # 1. the whole preconditioner against a host reference
    zb = ctx.buffer(n_f * 4)
    pre.apply(rb, zb)
    got = zb.download(np.float32, n_f).astype(np.float64)
    ref = np.zeros(n_f)
    for e in exts:
        d = diags[e]; fl = FLOOR_REL * d.max()
        if e == ext_f:
            ref += r / (d + fl)
        else:
            rc = multilinear_restrict(r, ext_f, e, nch)
            ref += multilinear_resize(rc / (d + fl), e, ext_f, nch)
    rel = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30))
    print(f"  C^-1 vs host reference: rel {rel:.2e}")
    assert rel < 1e-4, rel

    # 2. the transfers themselves, and the ADJOINT identity <Px,y> == <x,P'y>
    L = [x for x in pre.levels if x["T"] is not None][0]
    ec = L["ext"]; nc = int(np.prod(ec)) * nch
    xc = rng.normal(size=nc).astype(np.float32)
    cb = ctx.buffer(nc * 4); cb.upload(xc)
    fb = ctx.buffer(n_f * 4)
    L["T"].prolong(cb, fb)
    e1 = float(np.abs(fb.download(np.float32, n_f)
                      - multilinear_resize(xc, ec, ext_f, nch)).max()
               / (np.abs(multilinear_resize(xc, ec, ext_f, nch)).max() + 1e-30))
    Px = fb.download(np.float32, n_f)
    y = rng.normal(size=n_f).astype(np.float32)
    yb = ctx.buffer(n_f * 4); yb.upload(y)
    L["T"].restrict(yb, cb)
    Pty = cb.download(np.float32, nc)
    lhs = float(np.dot(Px.astype(np.float64), y.astype(np.float64)))
    rhs = float(np.dot(xc.astype(np.float64), Pty.astype(np.float64)))
    print(f"  prolong vs multilinear_resize: rel {e1:.2e}")
    print(f"  adjoint <Px,y> {lhs:.6f} vs <x,P'y> {rhs:.6f}")
    assert e1 < 1e-5, e1
    assert abs(lhs - rhs) <= 1e-4 * max(abs(lhs), 1.0), (lhs, rhs)

    # 3. accumulate: prolong(accum=True) adds rather than overwrites
    cb.upload(xc)
    fb.upload(np.ones(n_f, np.float32))
    L["T"].prolong(cb, fb, accum=True)
    acc = fb.download(np.float32, n_f)
    e3 = float(np.abs(acc - (Px + 1.0)).max() / (np.abs(Px).max() + 1e-30))
    print(f"  prolong accum: rel {e3:.2e}")
    assert e3 < 1e-5, e3
    print("BPX OK")
    ctx.destroy()


if __name__ == "__main__":
    main()
