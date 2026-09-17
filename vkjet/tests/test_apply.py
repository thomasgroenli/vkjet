"""The GPU tensor-product forward apply equals the numpy oracle, for 1D..4D
uniform cubic splines and a mixed-order case."""
import unittest

import numpy as np

from vkjet.apply import ApplyForward, apply_oracle
from vkjet.basis import Basis1D
from vkjet.tests import context


class TestApplyForward(unittest.TestCase):
    CASES = [  # (name, knots per dim, orders, channels, samples)
        ("1d-cubic", [range(9)], [4], 4, 1000),
        ("2d-cubic", [range(7), range(6)], [4, 4], 4, 800),
        ("3d-cubic", [range(5), range(6), range(5)], [4, 4, 4], 3, 500),
        ("4d-cubic", [range(5)] * 4, [4] * 4, 4, 300),
        ("mixed-order", [range(7), range(6), range(8)], [4, 2, 3], 2, 500),
    ]

    def test_gpu_equals_oracle(self):
        ctx = context()
        for seed, (name, knots, orders, nch, n) in enumerate(self.CASES):
            with self.subTest(case=name):
                rng = np.random.default_rng(seed)
                bases = [Basis1D.bspline(list(k), o) for k, o in zip(knots, orders)]
                ext = [b.primal_extent for b in bases]
                primal = rng.standard_normal(tuple(ext) + (nch,)).astype(np.float32)
                x_enc = np.stack([rng.uniform(0.0, e - 1e-3, n) for e in ext], 1).astype(np.float32)
                gpu = ApplyForward(ctx, bases, nch).forward(x_enc, primal)
                ref = apply_oracle(bases, x_enc, primal, nch)
                err = float(np.max(np.abs(gpu - ref)))
                self.assertLess(err, 2e-4, f"{name}: max |gpu - oracle| {err:.2e}")


if __name__ == "__main__":
    unittest.main()
