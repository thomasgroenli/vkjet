"""Vendored B-spline basis generator (from genspline, github/thomgronli).

vkjet ships its own copy so the package depends only on volkano + numpy.
This is a verbatim vendoring of genspline's Cox-de Boor machinery —
polynomial.py, knot_vector.py, basis.py, bspline.py — NOT a reimplementation:
the boundary behaviour of a finite knot vector (±inf ghost knots, degenerate
pieces) is subtle and the upstream code is the proven authority.

tests/test_bspline_vendor.py cross-checks this copy against upstream genspline
byte-for-byte whenever genspline is importable, so drift is detectable.

Do not edit these files to fix vkjet problems; fix them upstream and re-vendor.
"""
from .bspline import compute_bspline          # noqa: F401
from .knot_vector import KnotVector           # noqa: F401
