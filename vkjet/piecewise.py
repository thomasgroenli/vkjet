import math

import numpy as np

from .polynomial import Polynomial


class Pieces(dict):
    """`dict[int, Polynomial]` returning `Polynomial()` for missing piece indices.

    Lets look-forward storage stay sparse — degenerate intervals or pieces with
    no contribution simply stay out of the dict — while callers can still write
    `pieces[idx]` and get back a zero polynomial. Missing reads do **not**
    mutate the dict (unlike `defaultdict`), so `if not interval` still detects
    degeneracy and `interval.items()` still iterates only the live entries.
    """

    def __missing__(self, key):
        return Polynomial()
    
    def __setitem__(self, key, value):
        if value == Polynomial():
            if key in self:
                super().__delitem__(key)
        else:
            super().__setitem__(key, value)
    


class Basis:
    """Piecewise-polynomial basis over a periodic knot vector.

    `polynomials` is a list of length `kv.n_intervals`. Entry `polynomials[I]`
    is a `Pieces` (sparse `dict[int, Polynomial]`) mapping piece index `idx` ∈
    `[0, order)` to the polynomial of basis function `(I + idx) mod n_intervals`
    on interval `I` (**look-back** indexing — coefficient-buffer index `c` is
    the basis function whose support *ends* at interval `c`). Equivalently,
    piece `0` at interval `I` is the basis function whose support just
    started here (its first non-zero piece); piece `order - 1` is the basis
    function whose support is about to end. Polynomials are in interval-local
    coords `f ∈ [0, 1)`. Missing entries / empty intervals are zero
    contributions; degenerate (zero-width) intervals from padding carry empty
    `Pieces`.

    Use `dense()` to materialize the GPU-ready `(n_intervals, order, degree+1)`
    coefficient tensor.
    """

    def __init__(self, kv, polynomials):
        self.kv = kv
        self.polynomials = [Pieces(p) for p in polynomials]
        assert len(self.polynomials) == kv.n_intervals, \
            f"expected {kv.n_intervals} interval entries, got {len(self.polynomials)}"

    @property
    def n_intervals(self):
        return len(self.polynomials)

    @property
    def order(self):
        return max((max(d.keys(), default=-1) for d in self.polynomials), default=-1) + 1

    @property
    def degree(self):
        return max((p.degree for d in self.polynomials for p in d.values()), default=0)

    @property
    def dx(self):
        # Class-getitem trick: call as `basis.dx[k]`, not `basis.dx(k)`.
        parent = self
        class _:
            def __class_getitem__(cls, k):
                return parent._derivative(k)
        return _

    def _derivative(self, k):
        """k-th world-coord derivative (k > 0) or antiderivative (k < 0):
        apply `Polynomial.dx(k)` to each piece, then multiply by `width^-k`.
        The single `width^-k` formula handles both directions via the chain
        rule — world deriv = local deriv / width^k for k > 0; world antideriv
        = local antideriv * width^|k| for k < 0. Empty intervals are skipped
        so division-by-zero never arises at degenerate widths."""
        new_polys = [Pieces() for _ in range(self.n_intervals)]
        for i, interval in enumerate(self.polynomials):
            if not interval:
                continue
            scale = (self.kv[i + 1] - self.kv[i]) ** (-k)
            for idx, poly in interval.items():
                new_polys[i][idx] = poly.dx(k) * scale
        return Basis(self.kv, new_polys)

    def integral(self):
        """Order-`p+1` antiderivative basis with a coefficient cumsum transform.

        Returns `(antideriv_basis, transform)` such that for any coefficient
        buffer `c` and any sample `x`,

            Σ_idx antideriv_basis.polynomials[I][idx](f) ·
                  transform(c)[(I + idx) mod n]

        equals the antiderivative of the spline
        `Σ_idx self.polynomials[I][idx](f) · c[(I + idx) mod n]`, where
        `(I, f) = kv.encode(x)`.

        Construction. New basis fns are
        `Ñ_i^(p+1)(x) = A_{i-1}(x)/K_{i-1} − A_i(x)/K_i`, where
        `A_c(x) = ∫ N_c^(p)(t) dt` (zero before basis fn `c`'s support, `K_c`
        after) and `K_c` is the full integral of `N_c^(p)`. The K-normalization
        makes the post-support `1 − 1` cancel so each `Ñ_i^(p+1)` is compactly
        supported on `p+1` intervals. By construction
        `dÑ_i/dx = N_{i-1}/K_{i-1} − N_i/K_i`, which telescopes against a
        forward-cumsum coefficient transform.

        Coefficient transform. `c'[i+1] = c'[i] + c[i]·K[i]` (forward weighted
        cumsum). The free initial `c'[0]` parameterizes the integration
        constant. `transform(c, c0=0.0)` accepts it explicitly.

        Limit cases (K = 0 or K = ∞). A basis fn with `K[c] = 0` is
        identically zero (degenerate, e.g., a phantom basis fn at a wrap-
        adjacent unbounded interval); a basis fn with `K[c] = ∞` saturates
        on an unbounded interval (constant non-zero piece on a `±∞`-bounded
        interval). In both limits `A_c/K[c] → 0` (limit-correct for the
        K-normalized I-spline), so the construction folds the corresponding
        piece term to zero. The transform requires `c[i] = 0` whenever
        `K[i] = ∞`: a non-zero saturation coef makes the spline asymptote to
        a non-zero value, whose antiderivative grows linearly past the
        boundary and isn't representable on the same kv (the kernel reads
        `f = 0` only on unbounded intervals — constant terms only).

        Periodicity caveat. The transform produces a length-`n` `c'` buffer
        with `c'[n] = c'[0] + Σ c·K` (sum over finite-K basis fns).
        The kernel reads `c'` modulo `n`, so the antiderivative is genuinely
        periodic only when `Σ c·K = 0`. If not, the kernel result is correct
        on each interval individually but discontinuous at the wraparound by
        `Σ c·K`. With bilateral `±∞` ghosts this is moot — the two
        unbounded intervals are distinct in the kv and carry independent
        constants.
        """
        n = self.n_intervals
        p = self.order

        # Walk each basis fn's support intervals once, caching `R[I, idx]`
        # (cumulative integral of basis fn `(I+idx) mod n` up to `kv[I]`)
        # and `K[c]` (full integral). The look-back piece index at the
        # `o`-th support interval is `p - 1 - o`. A non-empty piece on a
        # non-finite-width interval pins `K[c] = ∞` (saturation).
        R = np.zeros((n, p))
        K = np.zeros(n)
        for c in range(n):
            running = 0.0
            for o in range(p):
                j = (c - p + 1 + o) % n
                old_idx = p - 1 - o
                R[j, old_idx] = running
                local_poly = self.polynomials[j][old_idx]
                if local_poly:
                    width_j = self.kv[j + 1] - self.kv[j]
                    if not math.isfinite(width_j):
                        running = float('inf')
                    else:
                        running += (local_poly.dx(-1) * width_j)(1)
            K[c] = running

        # New polynomial pieces: at interval I, idx_new ∈ [0, p+1) for basis
        # fn c_new = (I+idx_new) mod n. Piece is term_prev − term_new where
        # each term is `A_c|_I / K[c]`, with the K-limit collapses for
        # K[c] ∈ {0, ∞}. Active pieces: `A_c|_I(f) = R[I, idx] +
        # width(I)·∫P_{I, idx}`. Boundary cases:
        #   - idx_new = 0: c_prev just discharged → `A/K = 1` (when K finite).
        #   - idx_new = p: c_new not yet started → `A/K = 0`.
        # Width-`∞` intervals keep their pieces (the kernel evaluates them at
        # f=0); width-0 intervals are skipped (degenerate, no contribution).
        new_polynomials = [Pieces() for _ in range(n)]
        for I in range(n):
            width_I = self.kv[I + 1] - self.kv[I]
            if width_I == 0:
                continue

            for idx_new in range(p + 1):
                c_new = (I + idx_new) % n
                c_prev = (I + idx_new - 1) % n

                # term_new = A_{c_new}|_I / K[c_new].
                if idx_new == p or K[c_new] == 0 or not math.isfinite(K[c_new]):
                    term_new = Polynomial()
                else:
                    # K[c_new] finite & non-zero ⇒ if width=∞ the piece must
                    # be empty (else K would be ∞), so local_anti collapses.
                    local_poly = self.polynomials[I].get(idx_new, Polynomial())
                    local_anti = (local_poly.dx(-1) * width_I) if math.isfinite(width_I) else Polynomial()
                    A_cnew = Polynomial({0: R[I, idx_new]}) + local_anti
                    term_new = A_cnew * (1.0 / K[c_new])

                # term_prev = A_{c_prev}|_I / K[c_prev].
                if K[c_prev] == 0 or not math.isfinite(K[c_prev]):
                    term_prev = Polynomial()
                elif idx_new == 0:
                    term_prev = Polynomial({0: 1.0})
                else:
                    local_poly = self.polynomials[I].get(idx_new - 1, Polynomial())
                    local_anti = (local_poly.dx(-1) * width_I) if math.isfinite(width_I) else Polynomial()
                    A_cprev = Polynomial({0: R[I, idx_new - 1]}) + local_anti
                    term_prev = A_cprev * (1.0 / K[c_prev])

                new_polynomials[I][idx_new] = term_prev - term_new

        antideriv_basis = Basis(self.kv, new_polynomials)

        K_fixed = K.copy()

        def transform(c, c0=0.0):
            c = np.asarray(c, dtype=float)
            assert c.shape == (n,), \
                f"expected coefficient buffer of shape ({n},), got {c.shape}"
            # Saturation basis fns must have zero coefs at every index, not just
            # those swept by the cumsum recurrence — the n-1 entry never
            # enters c_prime but its non-zero c would still saturate the
            # spline on the unbounded interval.
            for i in range(n):
                if not math.isfinite(K_fixed[i]):
                    assert c[i] == 0.0, \
                        f"c[{i}] must be 0: K[{i}] is infinite (saturation basis fn) " \
                        f"and a non-zero coef makes the antiderivative grow linearly " \
                        f"past the unbounded interval, which the kernel cannot represent."
            c_prime = np.empty(n)
            c_prime[0] = c0
            for i in range(n - 1):
                if not math.isfinite(K_fixed[i]):
                    c_prime[i + 1] = c_prime[i]
                else:
                    c_prime[i + 1] = c_prime[i] + c[i] * K_fixed[i]
            return c_prime

        return antideriv_basis, transform

    def dense(self):
        """GPU-ready tensor of shape `(n_intervals, order, degree+1)`.

        Slot `[I, idx, c]` is the c-th coefficient (low-to-high power) of the
        polynomial for basis function `(I + idx) mod n_intervals` on interval
        `I`. Missing/empty `Pieces` entries materialize as zeros."""
        n = self.n_intervals
        order = self.order
        degree = self.degree
        tensor = np.zeros((n, order, degree + 1))
        for i, interval in enumerate(self.polynomials):
            for idx, poly in interval.items():
                tensor[i, idx, :] = poly.as_list(length=degree + 1)
        return tensor

    def __repr__(self):
        return f"Basis(n_intervals={self.n_intervals}, order={self.order}, degree={self.degree})"
