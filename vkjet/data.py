"""Real-data loading for SplineFlow: the splineflow unified measurement schema.

Unified record (flow/data.py): x(D world coords), d(C covector = √w·d̂),
s(√w·s_orig), nyquist. The data-fit residual is r = ⟨d, û⟩ − s.

`Axes` maps world coordinates onto a spline grid's encoded coordinates
(interval + fraction). NOTE: compute_bspline is periodic, so all
axes use a periodic basis here; splineflow makes space non-periodic. For the
interior of the flow (the vessel) this is a good approximation — the wrap
artifact lives at the domain boundary, which is outside the masked flow region.
"""
from __future__ import annotations

import os

import numpy as np


class Axes:
    """Per-dim world bounds [lo,hi] + grid intervals → encoded coords [0,n)."""

    def __init__(self, lo, hi, n_intervals, periodic, basis="bspline"):
        self.lo = np.asarray(lo, np.float64)
        self.hi = np.asarray(hi, np.float64)
        self.ext = [int(n) for n in n_intervals]
        self.periodic = list(periodic)
        self.basis = basis

    def encode(self, x_world: np.ndarray) -> np.ndarray:
        """(N,D) world → (N,D) encoded; periodic dims wrap, others clamp."""
        x = np.asarray(x_world, np.float64)
        u = (x - self.lo) / (self.hi - self.lo)        # [0,1] within bounds
        xe = u * np.array(self.ext, np.float64)
        for d in range(len(self.ext)):
            n = self.ext[d]
            if self.periodic[d]:
                xe[:, d] = np.mod(xe[:, d], n)
            else:
                xe[:, d] = np.clip(xe[:, d], 0.0, n - 1e-4)
        return np.ascontiguousarray(xe, dtype=np.float32)

    def bases(self):
        """Order-4 cubic B-spline per dim. A singleton dim (n_intervals == 1) gets the constant basis — one
        coefficient, ∂/∂dim ≡ 0 — so 2D+t data runs through the 4D machinery
        without phantom degrees of freedom along the dead axis."""
        from .basis import Basis1D
        return [Basis1D.bspline(list(range(n + 1)), 4) if n > 1
                else Basis1D.constant() for n in self.ext]

    @classmethod
    def from_npz(cls, path, n_intervals, periodic=(True, False, False, False)):
        ax = np.load(os.path.expanduser(path))
        keys = ["T", "X", "Y", "Z"]
        lo = [float(ax[k][0]) for k in keys]
        hi = [float(ax[k][-1]) for k in keys]
        return cls(lo, hi, n_intervals, periodic)


def _subsample(a, n_sub, seed):
    N = len(a)
    if n_sub and n_sub < N:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(N, n_sub, replace=False))
        return a[idx]
    return a[:]


def load_unified(path, n_sub=None, seed=0, return_nyquist=False):
    """Unified-schema measurements → (x_world(N,D), d(N,C), s(N,)).

    return_nyquist=True appends the per-row nyquist column (venc on wrapped
    velocity rows, 0 on wall/θ rows) as a 4th array."""
    a = _subsample(np.load(os.path.expanduser(path), mmap_mode="r"), n_sub, seed)
    out = (np.ascontiguousarray(a["x"], np.float32),
           np.ascontiguousarray(a["d"], np.float32),
           np.ascontiguousarray(a["s"], np.float32))
    if return_nyquist:
        ny = (np.ascontiguousarray(a["nyquist"], np.float32)
              if "nyquist" in a.dtype.names else np.zeros(len(a), np.float32))
        out = out + (ny,)
    return out


def save_unified(path, x, d, s, nyquist=0.0):
    """Write one unified measurement file (fields x, d, s, nyquist).

    nyquist: scalar or per-row (venc for wrapped velocity rows; 0 marks rows
    with no wrap physics — wall-image and θ rows)."""
    x = np.asarray(x, np.float32)
    d = np.asarray(d, np.float32)
    dt = np.dtype([("x", "<f4", (x.shape[1],)), ("d", "<f4", (d.shape[1],)),
                   ("s", "<f4"), ("nyquist", "<f4")])
    a = np.empty(len(x), dt)
    a["x"], a["d"], a["s"] = x, d, np.asarray(s, np.float32)
    a["nyquist"] = nyquist
    a["fuzz"] = fuzz
    np.save(os.path.expanduser(path), a)


def merge_rows(*rows):
    """Concatenate (x, d, s) row sets, zero-padding covectors to the widest
    channel count. One file, one schema: velocity, θ and wall-image rows mix
    freely; fit_flow5 classifies each row by its covector support."""
    C = max(np.asarray(r[1]).shape[1] for r in rows)
    xs, ds, ss = [], [], []
    for xr, dr, sr in rows:
        dr = np.asarray(dr, np.float32)
        dp = np.zeros((len(xr), C), np.float32)
        dp[:, :dr.shape[1]] = dr
        xs.append(np.asarray(xr, np.float32))
        ds.append(dp)
        ss.append(np.asarray(sr, np.float32))
    return np.concatenate(xs), np.concatenate(ds), np.concatenate(ss)


# --------------------------------------------------------------------------- #
# Jet-row schema: data AND equations as rows in ONE file (vkjet.eqrow)        #
# --------------------------------------------------------------------------- #
ROW_DTYPE = np.dtype([("x", "<f4", (4,)), ("op", "<i4"), ("w", "<f4"),
                      ("s", "<f4"), ("c", "<f4", (8,)), ("nyquist", "<f4"),
                      ("fuzz", "<f4")])
# nyquist: the row's modulus m (> 0 asserts the congruence r = 0 mod m — an
# aliased Doppler reading with m = 2*venc; 0 = the plain equality r = 0).
# fuzz: per-row sigma, in CELL units, of a Gaussian jitter applied to the row's
# evaluation point, redrawn once per optimiser step. A jittered row declares a
# MEASURE about x rather than a sampled point (physics rows: a fixed
# collocation set gets overfitted); data rows want fuzz = 0.


def make_rows(x, op, w, s, c=None, nyquist=0.0, fuzz=0.0):
    """Assemble a jet-row record array: point, operator id, absolute weight,
    RHS target, 8-float payload (per-row covectors etc.), nyquist."""
    x = np.asarray(x, np.float32)
    a = np.zeros(len(x), ROW_DTYPE)
    a["x"] = x
    a["op"] = np.asarray(op, np.int32)
    a["w"] = np.asarray(w, np.float32)
    a["s"] = np.asarray(s, np.float32)
    if c is not None:
        c = np.asarray(c, np.float32).reshape(len(x), -1)
        a["c"][:, :c.shape[1]] = c
    a["nyquist"] = nyquist
    return a


def save_rows(path, rows, ops):
    """ONE file = data rows + equation rows + the operator table (npz).

    Kernel hints are NOT serialized: the file is purely the objective
    (semantics + hash); execution routing is decided at fit time (JIT
    generation / dispatch), never by file metadata."""
    import json
    arrays = ops.to_arrays()
    arrays.pop("kernels", None)
    meta = json.dumps({"version": 1,
                       "nch": int(getattr(ops, "n_channels", 5)),
                       "slots": "jet2-v1",
                       "units": "world", "ncpr": 8})
    np.savez_compressed(os.path.expanduser(path), rows=np.asarray(rows),
                        meta_json=np.asarray(meta), **arrays)


def load_rows(path):
    """→ (rows record array, OperatorTable)."""
    from .eqrow import OperatorTable
    z = np.load(os.path.expanduser(path), allow_pickle=False)
    nch = 5                                    # files predating the field
    if "meta_json" in getattr(z, "files", []):
        import json
        nch = int(json.loads(str(z["meta_json"])).get("nch", 5))
    r = z["rows"]
    if "fuzz" not in (r.dtype.names or ()) or "nyquist" not in (r.dtype.names or ()):
        out = np.zeros(len(r), ROW_DTYPE)           # files written before a column
        for f in r.dtype.names:
            out[f] = r[f]
        r = out
    return r, OperatorTable.from_arrays(z, n_channels=nch)


def merge_row_sets(*sets):
    """Merge (rows, ops) sets into one; operators deduped by canonical form
    (name-independent), row op ids remapped."""
    from .eqrow import OperatorTable
    # a merged objective spans the widest channel count of its parts
    merged = OperatorTable(n_channels=max(
        int(getattr(o, "n_channels", 5)) for _, o in sets))
    keys = {}
    out_rows = []
    for rows, ops in sets:
        remap = np.empty(max(ops.n_ops, 1), np.int32)
        for k in range(ops.n_ops):
            ck = ops.canonical_key(k)
            if ck not in keys:
                keys[ck] = merged.add_op(ops.names[k], ops.lin[k],
                                         ops.quad[k], kernel=ops.kernels[k])
            remap[k] = keys[ck]
        r = np.asarray(rows).copy()
        if len(r):
            r["op"] = remap[r["op"]]
        out_rows.append(r)
    return np.concatenate(out_rows), merged


def data_operator(ops=None, name="data", n_channels=5, target_cix=None):
    """Register the shared value-covector operator: r = Σ_ch c[ch]·f_ch − s.
    Payload c carries the per-row covector (beam direction, wall normal,
    e_θ, √w·e_I, ...). Returns (ops, op_id)."""
    from .eqrow import OperatorTable, SLOT_VAL
    if ops is None:
        ops = OperatorTable()
    op_id = ops.add_op(name, lin=[(SLOT_VAL, ch, 1.0, ch)
                                  for ch in range(n_channels)],
                       const=(() if target_cix is None
                              else ((-1.0, int(target_cix)),)),
                       kernel="data" if target_cix is None else "")
    return ops, op_id


def rows_from_unified(x, d, s, nyquist=0.0, ops=None, target_cix=None):
    """Convert legacy unified rows (x, d(≤5), s) — √w folded into d and s —
    into jet rows referencing the shared data operator (w=1, covector in the
    payload; exactly the same residual). Returns (rows, ops)."""
    d = np.asarray(d, np.float32).reshape(len(x), -1)
    dp = np.zeros((len(x), 5), np.float32)
    dp[:, :d.shape[1]] = d
    ops, op_id = data_operator(ops, target_cix=target_cix)
    if target_cix is None:
        rows = make_rows(x, np.full(len(x), op_id, np.int32),
                         np.ones(len(x), np.float32), s, dp, nyquist)
    else:
        # target rides payload slot target_cix; the row's s column is 0 and the
        # residual is homogeneous (r = 0)
        cp = np.zeros((len(x), int(target_cix) + 1), np.float32)
        cp[:, :dp.shape[1]] = dp
        cp[:, int(target_cix)] = np.asarray(s, np.float32)
        rows = make_rows(x, np.full(len(x), op_id, np.int32),
                         np.ones(len(x), np.float32),
                         np.zeros(len(x), np.float32), cp, nyquist)
    return rows, ops


def rows_hash(rows, ops, fit_config=None):
    """Content hash of the OBJECTIVE an experiment minimizes.

    Canonical: op ids are replaced by the operator's canonical serialization
    (label- and id-independent), rows are sorted lexicographically (the
    objective is permutation-invariant), fit_config (grid ladder, steps, cg,
    seed, ...) is appended as sorted JSON. Same experiment ⇒ same hash,
    regardless of authoring order or operator numbering."""
    import hashlib
    import json
    rows = np.asarray(rows)
    keys = [repr(ops.canonical_key(k)).encode() for k in range(ops.n_ops)]
    opk = np.asarray([hashlib.sha256(k).hexdigest()[:16] for k in keys])
    canon = np.zeros(len(rows), dtype=[("k", "U16"), ("x", "<f4", (4,)),
                                       ("w", "<f4"), ("s", "<f4"),
                                       ("c", "<f4", (8,)), ("m", "<f4"),
                                       ("fz", "<f4")])
    canon["k"] = opk[rows["op"]]
    for f in ("x", "w", "s", "c"):
        canon[f] = rows[f]
    names = rows.dtype.names or ()
    # the modulus and the jitter ARE part of the objective (a congruence is
    # not an equality; a jittered row declares a measure, not a point)
    canon["m"] = rows["nyquist"] if "nyquist" in names else 0.0
    canon["fz"] = rows["fuzz"] if "fuzz" in names else 0.0
    canon = np.sort(canon, order=["k", "x", "w", "s", "m", "fz"])
    h = hashlib.sha256(canon.tobytes())
    # the channel count is part of the objective: the table references channel
    # indices, so the same rows on a different NCH are a different experiment
    h.update(f"|nch={int(getattr(ops, 'n_channels', 5))}".encode())
    if fit_config is not None:
        h.update(json.dumps(fit_config, sort_keys=True).encode())
    return h.hexdigest()
