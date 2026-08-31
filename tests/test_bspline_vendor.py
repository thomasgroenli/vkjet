"""The vendored B-spline generator must match upstream genspline exactly.

vkjet ships its own copy of genspline's Cox-de Boor machinery (vkjet/_gs) so
the package depends only on volkano + numpy. This test is the anti-drift gate:
whenever upstream genspline happens to be importable, every basis vkjet can
build must come out bit-identical. It SKIPS (not fails) when genspline is
absent, which is the normal state on a production machine.

Run: PYTHONPATH=~/projects/volkano python3 tests/test_bspline_vendor.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vkjet.basis import Basis1D                              # noqa: E402
from vkjet._gs import compute_bspline, KnotVector            # noqa: E402


def main():
    try:
        sys.path.insert(0, os.path.expanduser("~/genspline"))
        from genspline.bspline import compute_bspline as up_bspline
        from genspline.knot_vector import KnotVector as UpKV
    except Exception as ex:
        print(f"upstream genspline not importable ({type(ex).__name__}) — SKIP")
        return

    worst = 0.0
    n_cases = 0
    for order in (2, 3, 4, 5, 6):
        for n in (1, 2, 3, 4, 5, 6, 8, 10, 12, 20, 48, 80):
            knots = list(range(n + 1))
            try:
                up = up_bspline(UpKV(list(knots)), order).dense()
            except Exception:
                continue                       # upstream rejects it too -> skip
            mine = compute_bspline(KnotVector(list(knots)), order).dense()
            assert mine.shape == up.shape, f"order={order} n={n}: {mine.shape} vs {up.shape}"
            d = float(np.max(np.abs(np.asarray(mine, np.float64)
                                    - np.asarray(up, np.float64))))
            worst = max(worst, d)
            n_cases += 1
            assert d == 0.0, f"order={order} n={n}: max|diff|={d}"

    # non-uniform and negative knots too, since the vendored code supports them
    for knots in ([0, 1, 3, 7, 8], [-2, -1, 0, 2, 5], [0, 0.5, 0.75, 1.0, 3.0]):
        for order in (2, 3, 4):
            up = up_bspline(UpKV(list(knots)), order).dense()
            mine = compute_bspline(KnotVector(list(knots)), order).dense()
            d = float(np.max(np.abs(np.asarray(mine, np.float64)
                                    - np.asarray(up, np.float64))))
            worst = max(worst, d)
            n_cases += 1
            assert d == 0.0, f"knots={knots} order={order}: max|diff|={d}"

    # and the Basis1D wrapper the solver actually uses
    for n in (4, 6, 12, 48):
        b = Basis1D.uniform_cubic(n)
        up = up_bspline(UpKV(list(range(n + 1))), 4).dense()
        assert np.array_equal(b.dense, np.asarray(up, np.float32)), n
        assert b.primal_extent == n and b.stride == 1
        assert np.array_equal(b.table, np.arange(4, dtype=np.int32)[None, :])

    print(f"vendored generator == upstream genspline on {n_cases} bases "
          f"(max|diff| = {worst:g})")
    print("PASS")


if __name__ == "__main__":
    main()
