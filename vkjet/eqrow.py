"""Generic jet rows: arbitrary (≤ quadratic) PDEs as unified-schema rows.

A jet row generalizes the data row's covector-on-values to a covector on the
field JET at an explicit point x:

    r = ⟨L, J(f)(x)⟩ + Jᵀ Q J − s ,   loss += ½·scale·w·r²

J[slot][ch] = the NF=15 second-order jet (slot table below) of the NCH=5
field (u,v,w,θ,I). L is a sparse list of (slot, ch, val, cix) entries; Q a
sparse list of (slot1, ch1, slot2, ch2, val, cix) pairs. cix ≥ 0 multiplies
the entry by the row's 8-float payload c[cix] — per-row covectors (beam
directions, wall normals) ride one shared operator instead of per-row sparse
matrices. Data rows are the slot-0 special case; continuity/no-penetration
are linear with s = 0; NS advection and optical flow are quadratic; s ≠ 0
encodes sources and ALM shifts.

Units: operator coefficients are in WORLD units. EqRowTerm applies only the
chain rule (inv_widths = n_intervals/extent), so jet slots are true world
derivatives and one row file serves every grid of the dyadic ladder.

Slot table ("jet2-v1", dims ordered t,x,y,z):
    0 value · 1 ∂t · 2 ∂x · 3 ∂y · 4 ∂z · 5 ∂tt · 6 ∂xx · 7 ∂yy · 8 ∂zz ·
    9 ∂t∂x · 10 ∂t∂y · 11 ∂t∂z · 12 ∂x∂y · 13 ∂x∂z · 14 ∂y∂z
"""
from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np

from .context import Context, STORAGE
from .basis import Basis1D
from .apply import primal_strides
from .packing import (SHADER_DIR, MAX_NDIM, _value_coef_concat,
                      _table_concat, _deriv_coef_concat, _deriv2_coef_concat,
                      _coef_offsets, _table_offsets)
import os

EQROW_GRAD_SPV = os.path.join(SHADER_DIR, "eqrow_grad.spv")
EQROW_LOSS_SPV = os.path.join(SHADER_DIR, "eqrow_loss.spv")
EQROW_DIAG_SPV = os.path.join(SHADER_DIR, "eqrow_diag.spv")
EQROW_HVP_SPV = os.path.join(SHADER_DIR, "eqrow_hvp.spv")

NF = 15
NCH = 5
NCPR = 8                      # per-row payload floats
ROWREC = 5                    # ints per row record: op, w, s, fuzz, modulus
SLOT_VAL = 0
SLOT_DT, SLOT_DX, SLOT_DY, SLOT_DZ = 1, 2, 3, 4
SLOT_DTT, SLOT_DXX, SLOT_DYY, SLOT_DZZ = 5, 6, 7, 8
SLOT_DTX, SLOT_DTY, SLOT_DTZ = 9, 10, 11
SLOT_DXY, SLOT_DXZ, SLOT_DYZ = 12, 13, 14
# The ORDER-0 term of the same polynomial. A residual is
#     r = c0 + <L, J(f)> + J'QJ
# and the schema carried order-1 (lin) and order-2 (quad) as table entries while
# order-0 lived as a per-row `s` scalar with a hardcoded minus sign. SLOT_CONST
# is a linear entry with NO FIELD FACTOR, so the constant becomes just another
# term of the series: its value rides the payload like every other per-row
# coefficient (covectors, weights), and the residual is uniformly = 0. Slot 15
# is free in the pack format (slot<<8 | ch<<4 | cix+1, real slots 0..14), so
# this needs no new table section, no meta field and no ABI change.
SLOT_CONST = NF
MIXED_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
SLOT_NAMES = ["val", "dt", "dx", "dy", "dz", "dtt", "dxx", "dyy", "dzz",
              "dtx", "dty", "dtz", "dxy", "dxz", "dyz"]


class OperatorTable:
    """Sparse jet operators, deduplicated per file; rows reference by op id.

    add_op(name, lin=[(slot, ch, val[, cix])], quad=[(s1, c1, s2, c2, val
    [, cix])]) → op id.  cix ∈ [0, NCPR) multiplies the entry by the row
    payload c[cix]; omit or -1 for a constant coefficient. Q entries are NOT
    symmetrized — the gradient's product rule touches both sides."""

    def __init__(self, n_channels=NCH):
        # The table REFERENCES channel indices, so the objective it defines is
        # not well posed without a channel count — which makes n_channels part
        # of the semantic specification, carried by the row file alongside the
        # operators themselves (data.save_rows / load_rows), not a solver knob.
        self.n_channels = int(n_channels)
        self.names = []
        self.lin = []            # per op: list[(slot, ch, val, cix)]
        self.quad = []           # per op: list[(s1, c1, s2, c2, val, cix)]
        self.kernels = []        # per op: execution HINT ("" = none) — names
                                 # a fused implementation ("data",
                                 # "flow5-ns5/mom-u", ...). Routing metadata
                                 # only: NOT part of the operator's canonical
                                 # identity, NOT hashed, VERIFIED against the
                                 # coefficients at dispatch (mismatch → warn
                                 # + generic fallback).

    @property
    def n_ops(self):
        return len(self.names)

    def add_op(self, name, lin=(), quad=(), kernel="", const=()):
        """const: [(val, cix), ...] — order-0 terms. cix < 0 is a literal
        constant; cix >= 0 multiplies by the row's payload slot cix, which is
        how a per-row target/source is expressed without an `s` column."""
        L, Q = [], []
        for e in const:
            val = float(e[0]); cix = int(e[1]) if len(e) > 1 else -1
            assert -1 <= cix < NCPR
            L.append((int(SLOT_CONST), 0, val, cix))
        for e in lin:
            slot, ch, val = e[0], e[1], float(e[2])
            cix = int(e[3]) if len(e) > 3 else -1
            assert 0 <= slot <= NF and 0 <= ch < self.n_channels and -1 <= cix < NCPR
            L.append((int(slot), int(ch), val, cix))
        for e in quad:
            s1, c1, s2, c2, val = e[0], e[1], e[2], e[3], float(e[4])
            cix = int(e[5]) if len(e) > 5 else -1
            assert 0 <= s1 < NF and 0 <= c1 < self.n_channels
            assert 0 <= s2 < NF and 0 <= c2 < self.n_channels and -1 <= cix < NCPR
            Q.append((int(s1), int(c1), int(s2), int(c2), val, cix))
        self.names.append(str(name))
        self.lin.append(L)
        self.quad.append(Q)
        self.kernels.append(str(kernel))
        return len(self.names) - 1

    def canonical_key(self, k):
        """Name-independent identity of operator k (for merge dedup)."""
        return (tuple(sorted(self.lin[k])), tuple(sorted(self.quad[k])))

    def pack(self):
        """→ (optab_i, optab_f, n_ops, nnz_lin) GPU tables.

        optab_i = [lin_off(n_ops+1) | quad_off(n_ops+1) | lin_pack | quad_pack]
          lin_pack  = slot<<8 | ch<<4 | (cix+1)
          quad_pack = s1<<20 | c1<<16 | s2<<12 | c2<<8 | (cix+1)
        optab_f = [lin_val | quad_val]"""
        lin_off = np.cumsum([0] + [len(L) for L in self.lin]).astype(np.int32)
        quad_off = np.cumsum([0] + [len(Q) for Q in self.quad]).astype(np.int32)
        lp = [(s << 8) | (c << 4) | (cix + 1)
              for L in self.lin for (s, c, _, cix) in L]
        qp = [(s1 << 20) | (c1 << 16) | (s2 << 12) | (c2 << 8) | (cix + 1)
              for Q in self.quad for (s1, c1, s2, c2, _, cix) in Q]
        lv = [v for L in self.lin for (_, _, v, _) in L]
        qv = [v for Q in self.quad for (_, _, _, _, v, _) in Q]
        optab_i = np.concatenate([lin_off, quad_off,
                                  np.asarray(lp, np.int32),
                                  np.asarray(qp, np.int32)]).astype(np.int32)
        optab_f = np.asarray(lv + qv, np.float32)
        return optab_i, optab_f, self.n_ops, len(lv)

    # ---- (de)serialization for save_rows/load_rows ------------------------- #
    def to_arrays(self):
        lin_off = np.cumsum([0] + [len(L) for L in self.lin]).astype(np.int32)
        quad_off = np.cumsum([0] + [len(Q) for Q in self.quad]).astype(np.int32)
        f = lambda k: np.asarray([e[k] for L in self.lin for e in L], np.int32)
        g = lambda k: np.asarray([e[k] for Q in self.quad for e in Q], np.int32)
        return dict(
            lin_off=lin_off, lin_slot=f(0), lin_ch=f(1), lin_cix=f(3),
            lin_val=np.asarray([e[2] for L in self.lin for e in L], np.float32),
            quad_off=quad_off, quad_slot1=g(0), quad_ch1=g(1),
            quad_slot2=g(2), quad_ch2=g(3), quad_cix=g(5),
            quad_val=np.asarray([e[4] for Q in self.quad for e in Q],
                                np.float32),
            names=np.asarray(self.names),
            kernels=np.asarray(self.kernels))

    @classmethod
    def from_arrays(cls, a, n_channels=None):
        t = cls() if n_channels is None else cls(n_channels=int(n_channels))
        lo, qo = a["lin_off"], a["quad_off"]
        kern = (a["kernels"] if "kernels" in getattr(a, "files", a)
                else [""] * len(a["names"]))
        for k in range(len(a["names"])):
            lin = [(int(a["lin_slot"][e]), int(a["lin_ch"][e]),
                    float(a["lin_val"][e]), int(a["lin_cix"][e]))
                   for e in range(lo[k], lo[k + 1])]
            quad = [(int(a["quad_slot1"][e]), int(a["quad_ch1"][e]),
                     int(a["quad_slot2"][e]), int(a["quad_ch2"][e]),
                     float(a["quad_val"][e]), int(a["quad_cix"][e]))
                    for e in range(qo[k], qo[k + 1])]
            t.add_op(str(a["names"][k]), lin, quad, kernel=str(kern[k]))
        return t


# --------------------------------------------------------------------------- #
# numpy oracles (independent of the GPU path; float64)                        #
# --------------------------------------------------------------------------- #
def gather_fields_jet(bases, iw, x_enc, C):
    """(N, 15, NCH) second-order jet oracle (slot table above)."""
    db = [b.derivative(w) for b, w in zip(bases, iw)]
    ddb = [b.derivative(w).derivative(w) for b, w in zip(bases, iw)]
    nch = C.shape[-1]
    out = np.zeros((len(x_enc), NF, nch))
    order = [b.order for b in bases]
    for s_i, pt in enumerate(np.asarray(x_enc, np.float64)):
        I = np.floor(pt).astype(int)
        f = pt - I
        for offs in itertools.product(*[range(o) for o in order]):
            pv = np.empty(4); pd = np.empty(4); pq = np.empty(4)
            idxs = []
            for d, b in enumerate(bases):
                cr = I[d] % b.coef_period
                tr = I[d] % b.table_period
                wrap = (I[d] * b.stride + int(b.table[tr, offs[d]])) % b.primal_extent
                idxs.append(wrap)
                pv[d] = np.polynomial.polynomial.polyval(f[d], b.dense[cr, offs[d]])
                pd[d] = np.polynomial.polynomial.polyval(f[d], db[d].dense[cr, offs[d]])
                pq[d] = np.polynomial.polynomial.polyval(f[d], ddb[d].dense[cr, offs[d]])
            c = C[tuple(idxs)]
            W = np.zeros(NF)
            W[0] = pv.prod()
            for a in range(4):
                m = pv.copy(); m[a] = pd[a]; W[1 + a] = m.prod()
                m = pv.copy(); m[a] = pq[a]; W[5 + a] = m.prod()
            for m_i, (a, b_) in enumerate(MIXED_PAIRS):
                m = pv.copy(); m[a] = pd[a]; m[b_] = pd[b_]
                W[9 + m_i] = m.prod()
            out[s_i] += W[:, None] * c[None, :]
    return out


def soft_unwrap(r, m, tau):
    """Wrapped-Gaussian pseudo-residual r~ (float64 oracle of the GLSL
    softwrap): rows with m > 0 assert r ≡ 0 (mod m). Periodic form — r is
    wrapped to [-m/2, m/2] first, then the three branches k ∈ {-1, 0, 1} are
    marginalised at temperature sigma = tau·m (decoupled: the GN weight of the
    row is untouched, tau only softens the branch posterior). m <= 0 → r.
    tau < 0 selects the pure cosine loss (m/2π)²(1−cos(2πr/m)) instead."""
    r = np.asarray(r, np.float64).copy()
    m = np.broadcast_to(np.asarray(m, np.float64), r.shape)
    on = m > 0
    if not on.any():
        return r
    mm = m[on]
    if tau < 0:                       # pure cosine (first harmonic only)
        k = 2 * np.pi / mm
        r[on] = np.sin(k * r[on]) / k
        return r
    rr = r[on] - mm * np.round(r[on] / mm)
    s2 = np.maximum((tau * mm) ** 2, (1e-3 * mm) ** 2)
    u = np.stack([rr - mm, rr, rr + mm], 0)
    e = -0.5 * u * u / s2
    e -= e.max(0)
    w = np.exp(e)
    r[on] = (w * u).sum(0) / w.sum(0)
    return r


def wrapped_half_sq(r, m, tau):
    """Row loss oracle: ½·r² for m <= 0; for m > 0 the wrapped-Gaussian value
    -sigma²·log Σ_k exp(-u_k²/2sigma²) (wrapped first), whose derivative in r
    is exactly soft_unwrap and whose tau → 0 limit is ½·wrap(r)²."""
    r = np.asarray(r, np.float64)
    m = np.broadcast_to(np.asarray(m, np.float64), r.shape)
    out = 0.5 * r * r
    on = m > 0
    if on.any():
        mm = m[on]
        if tau < 0:
            k = 2 * np.pi / mm
            out = out.copy()
            out[on] = (1.0 - np.cos(k * r[on])) / (k * k)
            return out
        rr = r[on] - mm * np.round(r[on] / mm)
        s2 = np.maximum((tau * mm) ** 2, (1e-3 * mm) ** 2)
        u = np.stack([rr - mm, rr, rr + mm], 0)
        e = -0.5 * u * u / s2
        emax = e.max(0)
        out = out.copy()
        out[on] = -s2 * (emax + np.log(np.exp(e - emax).sum(0)))
    return out


def row_residuals_oracle(fields, ops, op, w, s, c):
    """fields (N,15,NCH) → residuals r (N,) per the operator table."""
    n = len(fields)
    r = -np.asarray(s, np.float64).copy()
    c = np.asarray(c, np.float64).reshape(n, -1)
    for i in range(n):
        k = int(op[i])
        for (slot, ch, val, cix) in ops.lin[k]:
            eff = val * (1.0 if cix < 0 else c[i, cix])
            r[i] += eff if slot == SLOT_CONST else eff * fields[i, slot, ch]
        for (s1, c1, s2, c2, val, cix) in ops.quad[k]:
            eff = val * (1.0 if cix < 0 else c[i, cix])
            r[i] += eff * fields[i, s1, c1] * fields[i, s2, c2]
    return r


def row_loss_oracle(bases, iw, x_enc, C, ops, op, w, s, c, scale=1.0,
                    modulus=None, tau=0.0):
    fields = gather_fields_jet(bases, iw, x_enc, np.asarray(C, np.float64))
    r = row_residuals_oracle(fields, ops, op, w, s, c)
    m = 0.0 if modulus is None else modulus
    return float(scale * np.sum(np.asarray(w, np.float64)
                                * wrapped_half_sq(r, m, tau)))


# --------------------------------------------------------------------------- #
# GPU term                                                                    #
# --------------------------------------------------------------------------- #
# 8 scalars + 9 per-dim tables + the wrapped-row temperature tau (float bits)
_META_EQ_INTS = 8 + 9 * MAX_NDIM + 1


def pack_eqrow_meta(bases, n_samples, scale, n_ops, nnz_lin, seed=0,
                    n_channels=NCH, tau=0.0):
    nd = len(bases)
    order = [b.order for b in bases]
    extents = [b.primal_extent for b in bases]
    pstride = primal_strides(extents, int(n_channels))
    m = np.zeros(_META_EQ_INTS, dtype=np.int32)
    m[0] = nd; m[1] = n_samples; m[2] = int(n_channels)
    m[3] = int(np.prod(order))
    m[4] = np.float32(scale).view(np.int32)
    m[5] = n_ops; m[6] = nnz_lin; m[7] = int(seed) & 0x7fffffff

    def put(slot, vals):
        base = 8 + slot * MAX_NDIM
        m[base:base + len(vals)] = vals
    put(0, [b.coef_period for b in bases]); put(1, order)
    put(2, [b.degp1 for b in bases]); put(3, [b.stride for b in bases])
    put(4, [b.table_period for b in bases]); put(5, extents)
    put(6, pstride); put(7, _coef_offsets(bases)); put(8, _table_offsets(bases))
    m[8 + 9 * MAX_NDIM] = np.float32(tau).view(np.int32)
    return m.tobytes()


class EqRowTerm:
    """Generic jet-row GN term (accumulate / accumulate_diag / hvp / loss).

    Everything the operator table can express — data rows, boundary
    conditions, linear PDEs, quadratic PDEs — through four generic kernels.
    Residual quadratic in c ⇒ loss exactly quartic (LM quartic line search
    stays exact) and the central-difference exact-Newton path is valid."""

    newton_fd = True

    def __init__(self, ctx: Context, bases: Sequence[Basis1D], inv_widths,
                 ops: OperatorTable):
        assert len(bases) == 4, "eqrow kernels are 4D"
        self.ctx = ctx
        self.bases = list(bases)
        self.inv_widths = list(inv_widths)
        self.ops = ops
        self.extents = [b.primal_extent for b in bases]
        self.nch = int(getattr(ops, "n_channels", NCH))
        self.primal_count = int(np.prod(self.extents)) * self.nch
        self.spec_stride = 1 if all(b.stride == 1 for b in bases) else 0x7fffffff
        self.spec_tper = 1 if all(b.table_period == 1 for b in bases) else 0
        self.grad_program = ctx.program(EQROW_GRAD_SPV, bindings=[STORAGE] * 12,
                                        spec_constant_ids=[0, 1, 2])
        self.loss_program = ctx.program(EQROW_LOSS_SPV, bindings=[STORAGE] * 12,
                                        spec_constant_ids=[0, 1, 2])
        self.diag_program = ctx.program(EQROW_DIAG_SPV, bindings=[STORAGE] * 12,
                                        spec_constant_ids=[0, 1, 2])
        self.hvp_program = ctx.program(EQROW_HVP_SPV, bindings=[STORAGE] * 13,
                                       spec_constant_ids=[0, 1, 2])
        self.coefs_buf = ctx.buffer(max(_value_coef_concat(self.bases).nbytes, 4))
        self.coefs_buf.upload(_value_coef_concat(self.bases))
        self.table_buf = ctx.buffer(max(_table_concat(self.bases).nbytes, 4))
        self.table_buf.upload(_table_concat(self.bases))
        dcoefs = _deriv_coef_concat(self.bases, self.inv_widths)
        self.dcoefs_buf = ctx.buffer(max(dcoefs.nbytes, 4))
        self.dcoefs_buf.upload(dcoefs)
        dd = _deriv2_coef_concat(self.bases, self.inv_widths)
        self.ddcoefs_buf = ctx.buffer(max(dd.nbytes, 4))
        self.ddcoefs_buf.upload(dd)
        oi, of, self.n_ops, self.nnz_lin = ops.pack()
        self.optab_i = ctx.buffer(max(oi.nbytes, 4)); self.optab_i.upload(oi)
        self.optab_f = ctx.buffer(max(of.nbytes, 4)); self.optab_f.upload(of)
        self._batch = None
        self.minibatches = []       # bucketed sub-batches (bind_buckets)

    def _make_batch(self, x_enc, op, w, s, c, scale=1.0, fuzz=None,
                    modulus=None):
        """Build a persistent device batch from host arrays. Split out of
        bind_batch so a bucketed row set can hold several of them (see
        bind_buckets) without the solver knowing."""
        ctx = self.ctx
        x_enc = np.ascontiguousarray(x_enc, np.float32)
        n = x_enc.shape[0]
        op = np.ascontiguousarray(op, np.int32).reshape(-1)
        w = np.ascontiguousarray(w, np.float32).reshape(-1)
        s = np.ascontiguousarray(s, np.float32).reshape(-1)
        assert op.size == n and w.size == n and s.size == n
        assert op.min(initial=0) >= 0 and op.max(initial=0) < self.n_ops
        cpad = np.zeros((n, NCPR), np.float32)
        if c is not None:
            c = np.asarray(c, np.float32).reshape(n, -1)
            cpad[:, :c.shape[1]] = c
        # row record {op, w, s, fuzz, m}: slot 3 carries the per-row jitter
        # sigma; slot 4 the wrap modulus m of the congruence r ≡ 0 (mod m)
        # (0 = exact row). Both ride the record so neither costs a binding.
        rr = np.zeros((n, ROWREC), np.int32)
        rr[:, 0] = op
        rr[:, 1] = w.view(np.int32)
        rr[:, 2] = s.view(np.int32)
        if fuzz is not None:
            fz = np.ascontiguousarray(fuzz, np.float32).reshape(-1)
            assert fz.size == n, (fz.size, n)
            rr[:, 3] = fz.view(np.int32)
        if modulus is not None:
            mm = np.ascontiguousarray(modulus, np.float32).reshape(-1)
            assert mm.size == n, (mm.size, n)
            rr[:, 4] = mm.view(np.int32)
        tau = float(getattr(self, "tau", 0.0))
        meta_b = pack_eqrow_meta(self.bases, n, scale, self.n_ops,
                                 self.nnz_lin, 0, n_channels=self.nch, tau=tau)
        xb = ctx.buffer(x_enc.nbytes); xb.upload(x_enc.reshape(-1))
        mb = ctx.buffer(len(meta_b), device_local=False); mb.upload(meta_b)
        rrb = ctx.buffer(rr.nbytes); rrb.upload(rr.reshape(-1))
        rcb = ctx.buffer(cpad.nbytes); rcb.upload(cpad.reshape(-1))
        return dict(n=n, xb=xb, mb=mb, rrb=rrb, rcb=rcb, scale=scale, seed=0,
                    tau=tau)

    def bind_batch(self, x_enc, op, w, s, c, scale=1.0, fuzz=None,
                   modulus=None):
        self._batch = self._make_batch(x_enc, op, w, s, c, scale, fuzz, modulus)
        self.minibatches = [self._batch]
        return self

    def bind_buckets(self, x_enc, op, w, s, c, buckets, scale=1.0, fuzz=None,
                     modulus=None):
        """Bind the full row set AND one sub-batch per bucket.

        `buckets` is a list of index arrays partitioning range(n) — the LOCAL
        view of a global bucket assignment, so minibatch k of every term refers
        to the same slice of the one row system. K=1 aliases the full batch
        (same object, no copy), which is what makes the full solve a literal
        special case rather than a parallel path.

        Costs a second copy of the row data on the device (the union batch plus
        the buckets) for K>1; at 64 B/row that is the price of arbitrary bucket
        membership without an offset field in the shader meta.
        """
        self.bind_batch(x_enc, op, w, s, c, scale, fuzz, modulus)
        if len(buckets) == 1:
            assert len(buckets[0]) == self._batch["n"], "K=1 must be the whole set"
            self.minibatches = [self._batch]
            return self
        sl = lambda a, ix: (None if a is None else np.asarray(a)[ix])
        self.minibatches = [
            self._make_batch(np.asarray(x_enc)[ix], sl(op, ix), sl(w, ix),
                             sl(s, ix), sl(c, ix), scale, sl(fuzz, ix),
                             sl(modulus, ix))
            for ix in buckets]
        return self

    def _all_batches(self):
        seen, out = set(), []
        for b in [self._batch] + list(getattr(self, "minibatches", ())):
            if b is not None and id(b) not in seen:
                seen.add(id(b)); out.append(b)
        return out

    def set_seed(self, seed):
        """Advance the jitter draw. Call ONCE PER OPTIMISER STEP, never per
        dispatch: every loss/grad/diag/hvp inside one step must see the same
        perturbed points or the line search and CG are solving different
        problems. A no-op for terms with no fuzzed rows."""
        for b in self._all_batches():
            if b.get("seed") == int(seed):
                continue
            b["seed"] = int(seed)
            self._upload_meta(b)

    def _upload_meta(self, b):
        b["mb"].upload(pack_eqrow_meta(self.bases, b["n"], b["scale"],
                                       self.n_ops, self.nnz_lin,
                                       b.get("seed", 0), n_channels=self.nch,
                                       tau=b.get("tau", 0.0)))

    def set_scale(self, scale):
        assert self._batch is not None, "call bind_batch first"
        for b in self._all_batches():
            b["scale"] = float(scale)
            self._upload_meta(b)

    def set_tau(self, tau):
        """Temperature of the wrapped rows' branch posterior, sigma = tau·m
        per row. Constant for now (no schedule); a no-op for rows with m = 0.
        Stored on the term so later binds inherit it."""
        self.tau = float(tau)
        for b in self._all_batches():
            if b.get("tau") == self.tau:
                continue
            b["tau"] = self.tau
            self._upload_meta(b)

    def _spec(self):
        return {0: self.spec_stride, 1: self.spec_tper, 2: self.nch}

    def _shared(self, coef_buf, b=None):
        b = self._batch if b is None else b
        return [b["mb"], b["xb"], coef_buf, self.coefs_buf, self.table_buf,
                self.dcoefs_buf, self.ddcoefs_buf, b["rrb"], b["rcb"],
                self.optab_i, self.optab_f]

    def accumulate(self, coef_buf, grad_buf, batch=None):
        b = self._batch if batch is None else batch
        self.ctx.run(self.grad_program, self._shared(coef_buf, b) + [grad_buf],
                     groups=(b["n"] + 255) // 256, spec=self._spec())

    def accumulate_diag(self, coef_buf, diag_buf, batch=None):
        b = self._batch if batch is None else batch
        self.ctx.run(self.diag_program, self._shared(coef_buf, b) + [diag_buf],
                     groups=(b["n"] + 255) // 256, spec=self._spec())

    def hvp(self, coef_buf, v_buf, out_buf, batch=None):
        b = self._batch if batch is None else batch
        self.ctx.run(self.hvp_program,
                     self._shared(coef_buf, b) + [v_buf, out_buf],
                     groups=(b["n"] + 255) // 256, spec=self._spec())

    def loss(self, coef_buf, loss_buf, batch=None):
        b = self._batch if batch is None else batch
        self.ctx.run(self.loss_program, self._shared(coef_buf, b) + [loss_buf],
                     groups=(b["n"] + 255) // 256, spec=self._spec())
