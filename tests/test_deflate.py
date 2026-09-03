"""Deflation must keep the preconditioner SYMMETRIC (CG breaks otherwise) and
must actually capture the modes the base preconditioner leaves slow.

Synthetic operator: H = diag(d) with a deliberately nasty spectrum, base
preconditioner = a POOR diagonal (deliberately mismatched) so some modes survive
relaxation and the coarse space has something to find."""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vkjet.context import Context, STORAGE                    # noqa: E402
from vkjet.deflate import Deflation                           # noqa: E402
from vkjet.optim import VEC_MADD_DIAG_SPV, VEC_PDIV_SPV       # noqa: E402


class _DiagBase:
    """z = r / (p + eps) — a deliberately mismatched diagonal preconditioner."""

    def __init__(self, ctx, n, p):
        self.ctx = ctx; self.n = n
        self.prog = ctx.program(VEC_PDIV_SPV, bindings=[STORAGE] * 4)
        self.meta = ctx.buffer(12, device_local=False)
        self.meta.upload(struct.pack("<i2f", n, 1e-6, 0.0))
        self.p = ctx.buffer(n * 4); self.p.upload(p.astype(np.float32))

    def apply(self, r, z):
        self.ctx.run(self.prog, [r, self.p, z, self.meta],
                     groups=min((self.n + 255) // 256, 4096))


def main():
    ctx = Context()
    n = 200_000
    rng = np.random.default_rng(0)
    # nasty spectrum: a bulk near 1 plus a handful of very small eigenvalues —
    # exactly the "a few slow modes" case deflation exists for
    d = np.abs(rng.normal(1.0, 0.2, n)) + 0.5
    d[:40] = np.linspace(1e-4, 1e-3, 40)
    db = ctx.buffer(n * 4); db.upload(d.astype(np.float32))
    madd = ctx.program(VEC_MADD_DIAG_SPV, bindings=[STORAGE] * 4)
    mmeta = ctx.buffer(12, device_local=False)
    mmeta.upload(struct.pack("<i2f", n, 1.0, 0.0))

    def hvp(v, out):                       # out += H·v
        ctx.run(madd, [v, db, out, mmeta], groups=min((n + 255) // 256, 4096))

    base = _DiagBase(ctx, n, np.ones(n))   # mismatched: ignores the small modes
    defl = Deflation(ctx, base, n, hvp, m=8, n_relax=10, seed=1)

    # 1. SYMMETRY — the property CG actually requires
    x = ctx.buffer(n * 4); y = ctx.buffer(n * 4)
    zx = ctx.buffer(n * 4); zy = ctx.buffer(n * 4)
    xv = rng.normal(size=n).astype(np.float32)
    yv = rng.normal(size=n).astype(np.float32)
    x.upload(xv); y.upload(yv)
    defl.apply(x, zx); defl.apply(y, zy)
    lhs = float(np.dot(zx.download(np.float32, n).astype(np.float64), yv))
    rhs = float(np.dot(xv.astype(np.float64), zy.download(np.float32, n)))
    rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)
    print(f"  symmetry <C r_x, r_y> {lhs:.6e} vs {rhs:.6e}  rel {rel:.2e}")
    assert rel < 1e-4, (lhs, rhs)

    # 2. the coarse operator must be SPD
    w = np.linalg.eigvalsh(defl.E)
    print(f"  W'HW eigenvalues {w.min():.3e} .. {w.max():.3e}")
    assert w.min() > 0, w

    # 3. the learned space must overlap the SLOW modes (first 40 coordinates),
    #    which is the whole point — random vectors would not.
    Wm = np.stack([defl.W[j].download(np.float32, n) for j in range(defl.m)], 1)
    mass_slow = float((Wm[:40] ** 2).sum() / (Wm ** 2).sum())
    frac = 40 / n
    print(f"  mass of W on the 40 slow coords: {mass_slow:.4f} "
          f"(random would give {frac:.2e}) -> {mass_slow/frac:.0f}x enrichment")
    assert mass_slow > 100 * frac, mass_slow
    print("DEFLATE OK")
    ctx.destroy()


if __name__ == "__main__":
    main()
