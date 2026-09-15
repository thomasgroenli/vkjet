import math

from .piecewise import Basis, Pieces
from .polynomial import Polynomial


def _add_factor(rhs_p, lhs_p):
    """Cox-de Boor additive factor `(rhs_p - x) / (rhs_p - lhs_p)` as a
    polynomial in `(x - kv[i])`, with limit-correct branches:

    - `lhs_p == rhs_p`: zero denominator, additive contribution is 0.
    - `rhs_p` non-finite: IEEE yields `±∞ / ±∞ = NaN` for the constant
      term; the asymptotic limit of the factor is the constant `1`.
    """
    if lhs_p == rhs_p:
        return Polynomial({})
    if not math.isfinite(rhs_p):
        return Polynomial({0: 1})
    return Polynomial({0: rhs_p, 1: -1}) / (rhs_p - lhs_p)


def _mul_factor(rhs, lhs):
    """Cox-de Boor multiplicative factor `(x - lhs) / (rhs - lhs)` as a
    polynomial in the recurrence's polynomial coordinate, with limit-correct
    branches:

    - `rhs == lhs`: zero denominator, factor is 0.
    - `lhs` non-finite: IEEE yields `∞ / ∞ = NaN`; with monotone knots and
      the in-range short-circuit in `B(i)` this fires only for `lhs = -∞`
      (when `B(i)` runs in absolute coords because `kv[i] = -∞`), where
      the asymptotic limit of the factor is `1`.

    `rhs` non-finite (with finite `lhs`) needs no branch: IEEE handles
    `finite / ∞ = 0` natively, and the polynomial collapses to zero.
    """
    if rhs == lhs:
        return Polynomial({})
    if not math.isfinite(lhs):
        return Polynomial({0: 1})
    return Polynomial({0: -lhs, 1: 1}) / (rhs - lhs)


def compute_bspline(kv, order):
    """Cox-de Boor B-spline basis of given `order` over a `KnotVector` `kv`.

    Returns a `Basis` of `order` per interval (each polynomial piece has
    degree `order - 1`), in look-back storage: at interval `I`, piece
    `idx` ∈ `[0, order)` is the polynomial of basis function
    `(I + idx) mod kv.n_intervals`. Storage is sparse (`Pieces`) — degenerate
    (zero-width or `+∞`-supported) intervals carry no entries.

    The recurrence runs in `u = x - ref` coordinates and the final `subs`
    step maps to interval-local `f ∈ [0, 1)`. `ref` is `kv[i]` when
    finite (numerically stable kv-relative form for the common case;
    keeps intermediate polynomial coefficients small) and `0` when
    `kv[i] = -∞` (absolute coords, avoiding the `-∞ - (-∞) = NaN`
    poisoning). The `kv[i] = +∞` case (look-back wraparound through the
    right `+∞` ghost) is handled by an early-return: the basis fn is
    identically 0 there.

    Boundary handling flows from how the kv is composed:

    - **Periodic** (default): plain `kv` — the kernel wraps `(I + idx) mod n`.
    - **I-spline** (monotone S-curve at the right tail, asymptote 1):
      `kv.right_pad_inf(order - 1)`. Cox-de Boor's linear factors degenerate
      to constants in the `+∞` limit (additive → 1, multiplicative → 0).
    - **Symmetric clamped + shared saturation**:
      `kv.left_pad(p).right_pad(p).right_pad_inf(1)`. Compressed clamped
      basis fns on both sides; `+∞` extent + wrap-to-unbounded encode →
      one shared "free" coef controls saturation past either boundary.
    - **Bilateral ghosts, two free saturation coefs**:
      `kv.left_pad(p).right_pad(p).left_pad_inf(1).right_pad_inf(1)`.
      Independent left and right saturation coefs.
    """
    assert order <= kv.n_intervals, (
        f"Cox-de Boor on a periodic kv requires order <= n_intervals "
        f"(got order={order}, n_intervals={kv.n_intervals}). For higher "
        f"orders, pad the kv (e.g., kv.right_pad(order - 1) or one of "
        f"the compositional recipes in the docstring)."
    )
    n = kv.n_intervals
    k = order - 1

    def B(i):
        """Cox-de Boor recurrence on the order+1 knots starting at kv[i]; returns
        a `Pieces` keyed by `j` (the j-th supporting interval of basis function
        `i`, which is interval `(i + j) mod n`).
        """
        # Look-back wraparound through the right-ghost +∞ region: support
        # starts at +∞, basis fn ≡ 0. Short-circuit before offsets get
        # NaN-tainted (`+∞ - +∞ = NaN`).
        if kv[i] == float('inf'):
            return Pieces({})

        # Phantom suppression: when extent = +∞ comes from a `-∞` ghost on
        # the left but `base[-1]` is finite (no explicit `+∞`), the
        # periodic extension at indices past `n_knots - 1` produces virtual
        # `+∞`s that don't correspond to any explicit kv structure. The
        # look-back wraparound recurrence picks these up as if they were
        # real ghost anchors, producing phantom step basis fns that break
        # partition-of-unity at the unbounded left interval. Suppress here
        # — the right-only mirror naturally degenerates these via multi-`+∞`
        # equality guards (explicit `+∞` matches periodic-extension `+∞`),
        # so this is just restoring the symmetry.
        if (i + order > kv.n_knots - 1
                and kv.extent == float('inf')
                and math.isfinite(kv.base[-1])):
            return Pieces({})

        # Hybrid reference: kv[i] when finite (kv-relative; numerically
        # stable), 0 when kv[i] = -∞ (absolute coords; avoids `-∞ - (-∞)
        # = NaN`). The factor helpers handle ±∞ inputs symbolically either way.
        ref = kv[i] if math.isfinite(kv[i]) else 0

        polys = Pieces({0: Polynomial({0: 1})})
        for r in range(k):
            for j in reversed(range(r + 1)):
                lhs   = kv[i + j]             - ref
                rhs   = kv[i + j     + k - r] - ref
                lhs_p = kv[i + j + 1]         - ref
                rhs_p = kv[i + j + 1 + k - r] - ref

                polys[j + 1] += polys[j] * _add_factor(rhs_p, lhs_p)
                polys[j]     *= _mul_factor(rhs, lhs)

        result = Pieces({})
        for j in range(order):
            result[j] = polys[j].subs(Polynomial({0: kv[i + j] - ref, 1: kv[i + j + 1] - kv[i + j]}))
        return result

    polynomials = [Pieces() for _ in range(n)]
    for i in range(n):
        for j, poly in B(i).items():
            polynomials[(i + j) % n][k - j] = poly

    return Basis(kv, polynomials)
