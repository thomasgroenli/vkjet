
import functools as ft
import itertools as it
import operator as op
import sys
from numbers import Number


TOLERANCE = sys.float_info.epsilon


class Commutator:
    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other + (-self)

    def __sub__(self, other):
        return self + (-other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (1 / other)

    def __neg__(self):
        return self * (-1)


def polynomial_input(f):
    @ft.wraps(f)
    def wrapper(self, other):
        if isinstance(other, Number):
            other = self.__class__({0: other})
        return f(self, other)
    return wrapper


class Polynomial(Commutator, dict):
    def __init__(self, c=None):
        if c is None:
            c = {}
        assert isinstance(c, dict), f"Not a dictionary, {c}"
        super().__init__()
        for k, v in c.items():
            self[k] = v

    def __missing__(self, key):
        return 0

    def __setitem__(self, key, value):
        if abs(value) < TOLERANCE:
            if key in self:
                super().__delitem__(key)
        else:
            super().__setitem__(key, value)

    @property
    def degree(self):
        return max(self.keys(), default=0)

    @polynomial_input
    def __add__(self, other):
        P = self.__class__()
        for (i, a) in it.chain(self.items(), other.items()):
            P[i] += a
        return P

    @polynomial_input
    def __mul__(self, other):
        P = self.__class__()
        for (i, a), (j, b) in it.product(self.items(), other.items()):
            P[i + j] += a * b
        return P

    @polynomial_input
    def longdiv(self, other):
        Q = self.__class__()
        R = self
        while R and R.degree >= other.degree:
            ratio = R[R.degree] / other[other.degree] * X ** (R.degree - other.degree)
            Q += ratio
            R -= ratio * other
        return Q, R

    @polynomial_input
    def __floordiv__(self, other):
        return self.longdiv(other)[0]

    @polynomial_input
    def __mod__(self, other):
        return self.longdiv(other)[1]

    def __pow__(self, power):
        if power >= 0:
            return ft.reduce(op.mul, it.repeat(self, power), Polynomial({0: 1}))
        return ft.reduce(op.floordiv, it.repeat(self, -power), Polynomial({0: 1}))

    def subs(self, poly):
        return sum((poly ** i * c for i, c in self.items()), self.__class__())

    def as_list(self, length=None):
        if length is None:
            length = self.degree + 1
        coefficients = [0] * length
        for k, v in self.items():
            coefficients[k] = v
        return coefficients

    def __call__(self, x):
        if isinstance(x, Number):
            return self._eval(x)
        return [self._eval(xi) for xi in x]

    def _eval(self, x):
        return sum(c * x ** i for i, c in self.items())

    def dx(self, order):
        if order == 0:
            return self
        if order > 0:
            P = Polynomial()
            for k, v in self.items():
                if k == 0:
                    continue
                P[k - 1] = k * v
            return P.dx(order - 1)
        # order < 0
        P = Polynomial()
        for k, v in self.items():
            if k == -1:
                raise ZeroDivisionError("Cannot integrate 1/x in power basis")
            P[k + 1] = (1 / (k + 1)) * v
        return P.dx(order + 1)


X = Polynomial({1: 1})
I = Polynomial({0: 1})
