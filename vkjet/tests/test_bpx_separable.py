"""The separable BPX transfer must BE the tensor-product operator, and Pᵀ its
exact adjoint — otherwise C⁻¹ is not symmetric and CG is not valid."""
import unittest

import numpy as np

from vkjet.bpx import BpxPreconditioner, FLOOR_REL
from vkjet.rowfit import multilinear_resize, multilinear_restrict
from vkjet.tests import context


class TestBpxSeparable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = context()
        cls.nch = 5
        cls.exts = [(4, 4, 4, 5), (8, 8, 8, 10), (16, 16, 16, 20)]
        cls.ext_f = cls.exts[-1]
        cls.n_f = int(np.prod(cls.ext_f)) * cls.nch
        cls.rng = np.random.default_rng(1)
        cls.diags, levels = {}, []
        for e in cls.exts:
            n = int(np.prod(e)) * cls.nch
            d = np.abs(cls.rng.normal(1.0, 0.3, n)) + 0.1
            cls.diags[e] = d
            buf = cls.ctx.buffer(n * 4); buf.upload(d.astype(np.float32))
            levels.append(dict(ext=e, diag=buf, dmax=float(d.max())))
        cls.pre = BpxPreconditioner(cls.ctx, cls.ext_f, cls.nch, levels)

    def test_preconditioner_vs_host_reference(self):
        r = self.rng.normal(size=self.n_f).astype(np.float32)
        rb = self.ctx.buffer(self.n_f * 4); rb.upload(r)
        zb = self.ctx.buffer(self.n_f * 4)
        self.pre.apply(rb, zb)
        got = zb.download(np.float32, self.n_f).astype(np.float64)
        ref = np.zeros(self.n_f)
        for e in self.exts:
            d = self.diags[e]; fl = FLOOR_REL * d.max()
            if e == self.ext_f:
                ref += r / (d + fl)
            else:
                rc = multilinear_restrict(r, self.ext_f, e, self.nch)
                ref += multilinear_resize(rc / (d + fl), e, self.ext_f, self.nch)
        self.assertLess(float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30)), 1e-4)

    def test_transfer_and_adjoint(self):
        L = [x for x in self.pre.levels if x["T"] is not None][0]
        ec = L["ext"]; nc = int(np.prod(ec)) * self.nch
        xc = self.rng.normal(size=nc).astype(np.float32)
        cb = self.ctx.buffer(nc * 4); cb.upload(xc)
        fb = self.ctx.buffer(self.n_f * 4)
        L["T"].prolong(cb, fb)
        ref = multilinear_resize(xc, ec, self.ext_f, self.nch)
        Px = fb.download(np.float32, self.n_f)
        self.assertLess(float(np.abs(Px - ref).max() / (np.abs(ref).max() + 1e-30)), 1e-5)
        y = self.rng.normal(size=self.n_f).astype(np.float32)
        yb = self.ctx.buffer(self.n_f * 4); yb.upload(y)
        L["T"].restrict(yb, cb)
        Pty = cb.download(np.float32, nc)
        lhs = float(np.dot(Px.astype(np.float64), y.astype(np.float64)))
        rhs = float(np.dot(xc.astype(np.float64), Pty.astype(np.float64)))
        self.assertLessEqual(abs(lhs - rhs), 1e-4 * max(abs(lhs), 1.0), (lhs, rhs))
        # accumulate: prolong(accum=True) adds rather than overwrites
        cb.upload(xc)
        fb.upload(np.ones(self.n_f, np.float32))
        L["T"].prolong(cb, fb, accum=True)
        acc = fb.download(np.float32, self.n_f)
        self.assertLess(float(np.abs(acc - (Px + 1.0)).max() / (np.abs(Px).max() + 1e-30)), 1e-5)


if __name__ == "__main__":
    unittest.main()
