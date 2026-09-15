import bisect
import math
import itertools as it
import operator as op
from numbers import Number


class KnotVector:
    """Ordered knot positions with periodic semantics.

    `KnotVector` is fundamentally periodic: `extent = base[-1] - base[0]`
    is the period. `get(i)` extends past the principal interval by adding
    one full `extent` per period in index space; `encode(x)` reduces
    world `x` modulo `extent` and guarantees a non-negative float index.

    Infinite extent (from `±∞` ghost knots) has two operating modes,
    determined by which sides carry ghosts:

    - **Right ghost only** (`base[-1] = +∞`, `base[0]` finite): the kv has
      one unbounded interval on the right. `encode(x < base[0])` wraps to
      that single unbounded interval — both sides past the principal
      share one "free" basis fn coefficient.

    - **Bilateral ghosts** (`base[0] = -∞` and `base[-1] = +∞`): the kv
      has two separate unbounded intervals. `encode` sends `x < base[1]`
      to the left unbounded interval (index `0`) and `x ≥ base[-2]` to
      the right unbounded interval (index `n_intervals - 1`). Two
      independent saturation coefficients — one per side.

    The single-`x < base[0]` branch in `encode` covers both cases: when
    `base[0] = -∞`, `x < -∞` is always False, so the branch is
    effectively conditional on the absence of a left ghost.

    Boundary behaviors flow from composition:
    - `left_pad(n)` / `right_pad(n)` repeat the boundary knot to compress
      basis-fn transitions there (knot multiplicity).
    - `left_pad_inf(n)` / `right_pad_inf(n)` extend by `±∞` ghost knots,
      adding unbounded intervals at the corresponding side.
    - Compose freely:
      - `kv.left_pad(p).right_pad(p).right_pad_inf(1)` — compressed
        clamped on both sides, single shared saturation coef.
      - `kv.left_pad(p).right_pad(p).left_pad_inf(1).right_pad_inf(1)` —
        compressed clamped on both sides, two independent saturation coefs.
    """

    def __init__(self, base):
        assert len(base) >= 2, "kv must have at least 2 knots"
        assert all(it.starmap(op.le, it.pairwise(base))), "knots must be ordered"
        assert base[-1] - base[0] > 0, \
            "extent must be positive (all-equal knots collapse encode's divmod; " \
            "all-±∞ kvs give NaN extent)"
        self.base = base

    @property
    def n_knots(self):
        return len(self.base)

    @property
    def n_intervals(self):
        return len(self.base) - 1

    @property
    def interval(self):
        return self.base[0], self.base[-1]

    @property
    def extent(self):
        return self.base[-1] - self.base[0]

    def locate(self, x):
        """Float index in `[0, n_intervals]` for `x` in the principal interval.

        Bisect + lerp. Returns `n + 0.0` on zero-width intervals (repeated
        knots) so the `0/0` doesn't propagate. For `+∞` knots, IEEE
        handles `lo finite, hi = +∞` natively (`finite/+∞ = 0` → `f = 0`,
        encoding stays in the unbounded interval). The remaining NaN case
        — `lo = +∞, hi = +∞` from a multiplicity of `+∞` ghosts — is
        indeterminate; falls back to `n + 0.0`. Out-of-range finite `x`
        extrapolates; callers (`encode`) clamp first when extent is `+∞`.
        """
        n = max(0, min(self.n_intervals - 1, bisect.bisect_right(self.base, x) - 1))
        lo, hi = self.base[n], self.base[n + 1]
        width = hi - lo
        if width == 0:
            return float(n)
        f = (x - lo) / width
        if math.isnan(f):
            return float(n)
        return n + f

    def get(self, i):
        """`base[r]` shifted by `q` full periods in index space.

        Standard formula `q * extent + base[r]` breaks at `q = 0` when
        `extent` is non-finite (`0 * ∞ = NaN`), so the in-range case
        short-circuits to `base[i]` directly — that *is* the `q = 0`
        limit. For `q ≠ 0` with `extent = +∞`, the shift dominates and
        the result is the relevant `±∞` — the math-correct limit of a
        periodic shift across an unbounded period.
        """
        n = self.n_intervals
        if 0 <= i < self.n_knots:
            return self.base[i]
        q, r = divmod(i, n)
        shift = q * self.extent
        if not math.isfinite(shift):
            return shift
        return shift + self.base[r]

    def encode(self, x):
        """Encode world `x` to a non-negative float index for the GPU.

        Integer part = interval index, fractional part = offset in `[0, 1)`.
        For finite `extent`, applies a positive period shift if `x` is
        below the principal interval, then reduces modulo `extent` —
        standard periodic wrap.

        For `extent = +∞`, the periodic structure wraps once at the
        cyclic seam, located in the unbounded interval. Two symmetric
        wrap branches handle the two ghost-ghost recipes:

        - `x < base[0]`: fires only when `base[0]` is finite (i.e., the
          kv has a `+∞` ghost on the right). Wraps to the right unbounded
          interval at index `n_intervals - 1`.
        - `x >= base[-1]`: fires only when `base[-1]` is finite (i.e., the
          kv has a `-∞` ghost on the left, no explicit `+∞` on the right).
          Wraps to the left unbounded interval at index `0`.

        For bilateral kvs (both `±∞` ghosts), neither branch fires
        (`x < -∞` and `x >= +∞` are both False for finite `x`), so encode
        falls through to `locate`, which lands x in the appropriate
        unbounded interval based on bisect_right.
        """
        if math.isfinite(self.extent):
            a = self.base[0]
            if x < a:
                n_periods = math.ceil((a - x) / self.extent)
                x = x + n_periods * self.extent
            q, r = divmod(x - a, self.extent)
            return q * self.n_intervals + self.locate(r + a)
        if x < self.base[0]:
            return float(self.n_intervals - 1)
        if x >= self.base[-1]:
            return 0.0
        return self.locate(x)

    def left_pad(self, n):
        """Return a new KnotVector with `n` extra copies of `base[0]` prepended."""
        return KnotVector([self.base[0]] * n + list(self.base))

    def right_pad(self, n):
        """Return a new KnotVector with `n` extra copies of `base[-1]` appended."""
        return KnotVector(list(self.base) + [self.base[-1]] * n)

    def left_pad_inf(self, n):
        """Return a new KnotVector with `n` `-∞` ghost knots prepended.

        Adds a left unbounded interval (index 0). With `right_pad_inf` on
        the other side, encoding handles the two unbounded intervals
        independently — `x < base[1]` lands at index 0, `x ≥ base[-2]`
        at index `n_intervals - 1` — giving two free saturation coefs.
        """
        return KnotVector([float('-inf')] * n + list(self.base))

    def right_pad_inf(self, n):
        """Return a new KnotVector with `n` `+∞` ghost knots appended."""
        return KnotVector(list(self.base) + [float('inf')] * n)

    def __getitem__(self, item):
        if isinstance(item, Number):
            i = math.floor(item)
            f = item - i
            if f == 0:
                return self.get(i)
            return (1 - f) * self.get(i) + f * self.get(i + 1)

        elif isinstance(item, slice):
            start, stop, step = item.start, item.stop, item.step
            assert None not in (start, stop, step), "slice requires explicit start, stop, and step"
            n = math.ceil((stop - start) / step)
            return [self[start + i * step] for i in range(n)]

        else:
            raise TypeError(str(item))

    def __len__(self):
        return self.n_knots

    def __repr__(self):
        return f"KnotVector(n_knots={self.n_knots}, interval={self.interval})"
