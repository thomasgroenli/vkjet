"""1D spline bases in the ``gs_basis_1d`` representation.

A :class:`Basis1D` carries exactly the fields the apply kernel's v5 addressing
consumes (shaders/kernel_apply.comp.glsl):

    coefficients : (coef_period, order, degp1) float32, low-to-high power
    stride       : signed int I-multiplier (1 for LOOKBACK B-splines)
    table        : (table_period, order) int32 offsets (identity for LOOKBACK)
    primal_extent: wrap modulus = n_intervals

The per-(sample, combo) weight/index for dim d is

    coef_row  = ix_int % coef_period
    table_row = ix_int % table_period
    sum       = ix_int*stride + table[table_row, idx]
    wrap      = sum % primal_extent
    weight   *= horner(coefficients[coef_row, idx, :], ix_frac)   # low->high
    prim     += wrap * pstride[d]

The polynomials come from the Cox-de Boor generator in :mod:`vkjet.bspline`
over a :class:`vkjet.knot_vector.KnotVector`.
"""
from __future__ import annotations

import numpy as np

from .bspline import compute_bspline
from .knot_vector import KnotVector


class Basis1D:
    def __init__(self, dense: np.ndarray, primal_extent: int,
                 stride: int = 1, table: np.ndarray = None):
        # dense: (coef_period, order, degp1), low-to-high power
        self.dense = np.ascontiguousarray(dense, dtype=np.float32)
        self.coef_period, self.order, self.degp1 = self.dense.shape
        self.primal_extent = int(primal_extent)
        self.stride = int(stride)
        if table is None:  # LOOKBACK identity: offsets [0, 1, ..., order-1]
            table = np.arange(self.order, dtype=np.int32)[None, :]
        self.table = np.ascontiguousarray(table, dtype=np.int32)
        self.table_period = self.table.shape[0]

    @classmethod
    def bspline(cls, knots, order: int) -> "Basis1D":
        """A B-spline of the given order over ``knots`` (LOOKBACK layout).

        ``knots`` may be a KnotVector or a sequence of knot positions.
        primal_extent = n_intervals = len(knots) - 1.
        """
        kv = knots if isinstance(knots, KnotVector) else KnotVector(list(knots))
        basis = compute_bspline(kv, order)
        dense = basis.dense()                       # (n_intervals, order, degp1)
        return cls(dense, primal_extent=basis.n_intervals)

    @classmethod
    def uniform_cubic(cls, n_intervals: int) -> "Basis1D":
        """Uniform cubic (order=4) B-spline on integer knots [0..n_intervals]."""
        return cls.bspline(list(range(n_intervals + 1)), order=4)

    @classmethod
    def constant(cls) -> "Basis1D":
        """Singleton dim: ONE coefficient, weight = 1, zero derivative.

        Collapses a dimension out of the tensor product (2D+t data through the
        4D machinery: give the dead axis this basis and the field is constant
        across it, d/d(axis) = 0 exactly via the all-zero derivative
        polynomial)."""
        return cls(np.ones((1, 1, 1), np.float32), primal_extent=1)

    def derivative(self, inv_width: float = 1.0) -> "Basis1D":
        """Basis whose polynomials are d/df of this one, scaled by inv_width.

        For p(f) = sum_k c_k f^k, p'(f) = sum_k k*c_k f^(k-1); the (1/width)
        chain rule (encoded f = (x-knot)/width) folds in as ``inv_width``. Same
        addressing (stride/table/primal_extent) - only the coefficients differ,
        so it gathers d_d fields through the identical support window.
        """
        D = self.degp1
        dcoef = np.zeros_like(self.dense)
        k = np.arange(1, D, dtype=np.float32)              # powers 1..D-1
        dcoef[:, :, :D - 1] = self.dense[:, :, 1:D] * k    # k*c_k -> coeff k-1
        dcoef *= np.float32(inv_width)
        return Basis1D(dcoef, self.primal_extent, self.stride, self.table)
