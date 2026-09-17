"""The object form of the contract: a Field (the unknown), a RowSystem (the
objective: row families on that field) and a Solve (convergence devices).

    field  = Field(lo, hi, n_channels=5, cell=(30e-3, 1e-3, None, 1e-3))
    system = RowSystem(field).add(family_a).add(family_b)
    res    = system.fit(Solve(bpx=True, steps=50))

A family is any object with a name and a build(system) -> (rows, ops) method;
families whose weights are the REFERENCE for others (a data family) set
reference = True and are built first, so a physics family can ask
system.reference_weight for the total data weight when it builds. A system is
STATIC: build() runs once; masses changed afterwards make a new system with a
new hash. The only continuation offered is a warm start (fit(init=...)).
Nothing solver-side lives on the system, and nothing objective-side (masses,
weights, moduli, the wrapped-loss temperature) lives on the Solve.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict

import numpy as np

from .apply import ApplyForward
from .context import Context
from .data import Axes, merge_row_sets, save_rows, load_rows, rows_hash
from .eqrow import NCH
from .rowfit import fit_rows


class Field:
    """The unknown: an n_channels spline field on a box, on a dyadic ladder of
    grids whose finest level is the solve grid. Resolution either as a base
    grid (per axis, coarsest level) or as finest cell sizes in WORLD units
    (`cell`, per axis, None keeps the base grid entry), rounded up to the
    ladder so the cell is never coarser than asked. A singleton axis (grid 1)
    is the constant basis."""

    def __init__(self, lo, hi, n_channels=NCH, base_grid=(9, 10, 1, 11), n_stages=4,
                 cell=None, periodic=(True, False, False, False)):
        self.lo = np.asarray(lo, np.float64); self.hi = np.asarray(hi, np.float64)
        self.n_channels = int(n_channels); self.n_stages = int(n_stages)
        self.periodic = tuple(periodic)
        base = list(int(b) for b in base_grid)
        if cell is not None:
            f = 1 << (self.n_stages - 1)
            for ax, c in enumerate(cell):
                if c is not None and c > 0:
                    base[ax] = int(np.ceil((self.hi[ax] - self.lo[ax]) / float(c) / f))
        self.base_grid = tuple(base)
        self.grids = [tuple(g * (1 << k) if g > 1 else 1 for g in self.base_grid)
                      for k in range(self.n_stages)]

    @property
    def grid(self):
        return self.grids[-1]

    @property
    def extent(self):
        return self.hi - self.lo

    @property
    def cells(self):
        return tuple(float(e / g) for e, g in zip(self.extent, self.grid))

    @property
    def n_coef(self):
        return int(np.prod(self.grid)) * self.n_channels

    def axes(self):
        return Axes(self.lo, self.hi, self.grid, periodic=self.periodic)

    def describe(self):
        c = self.cells
        return (f"grid {self.grid} on box t[{self.lo[0]:.3g},{self.hi[0]:.3g}] "
                f"x[{self.lo[1]:.3g},{self.hi[1]:.3g}] z[{self.lo[3]:.3g},{self.hi[3]:.3g}]: "
                f"cells {1e3*c[0]:.1f} ms, {1e3*c[1]:.2f} mm, {1e3*c[3]:.2f} mm; "
                f"{self.n_coef:,} coefficients, {self.n_channels} channels")


@dataclass
class Solve:
    """Convergence devices — none of them may shape the answer."""
    bpx: bool = True
    steps: int = 50
    cg_iters: int = 12
    bpx_floor: float = 1e-2
    stop_rel: float = 0.0
    stop_window: int = 5
    seed: int = 0
    gpu: int = 0
    verbose: bool = True


class FitResult:
    """Coefficients on the field + the solve's diagnostics + an evaluator."""

    def __init__(self, system, coef, losses, diagnostics, ctx):
        self.system = system; self.field = system.field
        self.coef = np.ascontiguousarray(coef, np.float32)
        self.losses = list(losses); self.diagnostics = diagnostics
        self._ctx = ctx; self._af = None
        self.loss = float(losses[-1])

    @property
    def steps_used(self):
        return self.diagnostics.get("steps_used")

    @property
    def trace(self):
        return self.diagnostics.get("pred_rel", [])

    def __call__(self, x_world):
        """(N, 4) world points -> (N, n_channels)."""
        f = self.field
        if self._af is None:
            self._axes = f.axes(); self._bases = self._axes.bases()
            ext = tuple(b.primal_extent for b in self._bases)
            self._af = ApplyForward(self._ctx, self._bases, f.n_channels)
            self._C = self.coef.reshape(ext + (f.n_channels,))
        return self._af.forward(self._axes.encode(np.asarray(x_world, np.float32)), self._C)

    def as_fit(self):
        """The dict the cardiacrows diagnostics accept."""
        return dict(coef=self.coef, lo=self.field.lo, hi=self.field.hi, grid=self.field.grid,
                    t0=getattr(self.field, "t0", None), frame=1)

    def save_summary(self, out):
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "fit.json"), "w") as f_:
            json.dump(dict(loss=self.loss, losses=self.losses, steps_used=self.steps_used,
                           floor=self.diagnostics.get("floor"),
                           pred_rel=[None if not np.isfinite(v) else float(v) for v in self.trace],
                           system=self.system.hash, field=self.field.describe()), f_, indent=1)

    def close(self):
        if self._ctx is not None:
            self._ctx.destroy(); self._ctx = None


class RowSystem:
    """The objective: an ordered list of row families on one field."""

    def __init__(self, field: Field):
        self.field = field
        self.families = []
        self._rows = self._ops = None
        self._built = {}                  # family name -> (rows, ops)

    # ---- authoring --------------------------------------------------------
    def add(self, family):
        assert self._rows is None, "the system is built; a change is a new system"
        assert not any(f.name == family.name for f in self.families), f"duplicate family {family.name}"
        self.families.append(family)
        return self

    def remove(self, name):
        assert self._rows is None, "the system is built; a change is a new system"
        self.families = [f for f in self.families if f.name != name]
        return self

    def family(self, name):
        for f in self.families:
            if f.name == name:
                return f
        raise KeyError(name)

    # ---- build ------------------------------------------------------------
    @property
    def reference_weight(self):
        """Total weight of the reference (data) families built so far."""
        return float(sum(float(r["w"].sum()) for f in self.families
                         if getattr(f, "reference", False) and f.name in self._built
                         for r, _ in [self._built[f.name]]))

    def build(self, verbose=True):
        if self._rows is not None:
            return self._rows, self._ops
        t0 = time.time()
        order = sorted(self.families, key=lambda f: 0 if getattr(f, "reference", False) else 1)
        for f in order:
            self._built[f.name] = f.build(self)
            if verbose:
                r, _ = self._built[f.name]
                print(f"  [{f.name}] {len(r):,} rows, weight {float(r['w'].sum()):,.0f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
        sets = [self._built[f.name] for f in self.families if self._built[f.name][0] is not None
                and len(self._built[f.name][0])]
        self._rows, self._ops = merge_row_sets(*sets)
        return self._rows, self._ops

    @property
    def rows(self):
        return self.build(verbose=False)[0]

    @property
    def ops(self):
        return self.build(verbose=False)[1]

    @property
    def hash(self):
        return rows_hash(self.rows, self.ops)

    def summary(self):
        self.build(verbose=False)
        ref = self.reference_weight
        lines = [self.field.describe()]
        for f in self.families:
            r, o = self._built[f.name]
            w = float(r["w"].sum()) if r is not None and len(r) else 0.0
            lines.append(f"  {f.name:16s} {len(r):>10,} rows  weight {w:>12,.0f}"
                         + (f"  ({w/ref:.3g} x reference)" if ref > 0 and not getattr(f, 'reference', False) else "")
                         + (f"  {f.describe()}" if hasattr(f, "describe") else ""))
        lines.append(f"  total {len(self.rows):,} rows, {self.ops.n_ops} operators, hash {self.hash[:16]}")
        return "\n".join(lines)

    def save(self, path):
        save_rows(path, self.rows, self.ops)
        return path

    # ---- solve ------------------------------------------------------------
    @property
    def tau(self):
        """The wrapped rows' temperature, declared by a family (e.g. Doppler)."""
        taus = {float(getattr(f, "tau")) for f in self.families if hasattr(f, "tau")}
        assert len(taus) <= 1, f"families disagree on tau: {taus}"
        return taus.pop() if taus else 0.0

    def fit(self, solve: Solve = None, init=None, ctx=None):
        S = solve or Solve()
        rows, ops = self.build(verbose=S.verbose)
        f = self.field
        own = ctx is None
        ctx = ctx or Context(device_index=int(S.gpu))
        res = fit_rows(rows, ops, lo=f.lo, hi=f.hi, base_grid=f.base_grid, n_stages=f.n_stages,
                       periodic=f.periodic, steps=S.steps, cg_iters=S.cg_iters, bpx=S.bpx,
                       bpx_floor=S.bpx_floor, tau=self.tau, init=init, seed=S.seed,
                       stop_rel=S.stop_rel, stop_window=S.stop_window, ctx=ctx, verbose=S.verbose)
        return FitResult(self, res.coef, res.stage_losses, res.diagnostics, ctx if own else ctx)


def load_system(path, field):
    """A saved row file as a RowSystem with one opaque family (the file)."""
    rows, ops = load_rows(path)

    class _File:
        name = "file"; reference = True
        def build(self, system): return rows, ops
    return RowSystem(field).add(_File())
