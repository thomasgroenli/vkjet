"""Row authoring: think in (w, s, covector); hand the solver (x, op, payload, m, fuzz).

The solver's contract is the smallest thing that defines the objective. A row's
weight is not part of it: if every coefficient of an operator rides a payload
slot, the residual is homogeneous of degree 1 in the payload, and

    w · r(c)²  =  r(√w · c)²

exactly. So the weight folds into the payload and the `w` column stops being
part of the wire format. The same is true of `s`: a per-row target riding
SLOT_CONST is just another coefficient, which is what SLOT_CONST exists for.

Wrapped rows fold too, with their modulus. A row with m > 0 asserts
r ≡ 0 (mod m) and is scored by the wrapped Gaussian at temperature σ = τ·m
(eqrow.wrapped_half_sq). Scaling r by √w and m by √w scales σ by √w as well,
and

    w · L(r; m, τ)  =  L(√w·r; √w·m, τ)

exactly, for the wrapped Gaussian, the τ → 0 sawtooth and the τ < 0 cosine
alike: every one of them is homogeneous of degree 2 in (r, m). The fold
therefore scales the `nyquist` column by √w along with the payload; a row's
modulus is part of its content, not a solver setting. `fuzz` is in cell units
of x and is untouched.

That leaves (x, op, payload, m, fuzz). But "weight" and "target" are how an
author actually thinks — confidence, calibration, IRLS, λ knobs, stratified
quadrature — so the vocabulary belongs HERE, on the authoring side, and the
folding happens on the way out. Nothing here touches the GPU.

THE FOLDED FORM IS NOT WRITE-HOSTILE. Scaling a row's payload and modulus by
√k scales its weight by k, so reweighting stays a natural operation after
folding (`reweight`); an author need not keep an unfolded copy to change their
mind about confidence.

WHAT IS GIVEN UP, stated plainly: after folding, w is not recoverable from a
row whose payload also carries direction — you see √w·c and cannot factor it.
Where an operator has a dedicated gain slot (physics rows with literal
coefficients) w = c[gain]² exactly, and `gain_of` returns it. Where it does
not (data rows, whose payload IS the covector), the weight is gone as a
separate quantity. The objective never used it separately; diagnostics that
did must read it before the fold.

Precision: √w is stored in a float32 payload, so w round-trips to ~1e-7
relative rather than exactly — an order below the fp32 atomic noise of the
reduction it feeds, but a property of the folded form, not an identity.

The solver's per-operator re-author diff (rowfit) recognises a common weight
factor on the `w` column and turns it into a term scale without rebinding;
folded rows carry that factor in the payload, so a reweight after folding is a
content change and rebinds. Fold once, on the way out.
"""
from __future__ import annotations

import numpy as np

from .eqrow import NCPR, SLOT_CONST, OperatorTable


def slots_used(ops, k):
    """Payload slots operator k references."""
    return {e[-1] for e in list(ops.lin[k]) + list(ops.quad[k]) if e[-1] >= 0}


def is_homogeneous(ops, k):
    """True if every coefficient of operator k rides a payload slot, i.e. r is
    linear in the payload and the weight can fold into it."""
    return all(e[-1] >= 0 for e in list(ops.lin[k]) + list(ops.quad[k]))


def free_slot(ops, k, taken=()):
    """A payload slot operator k does not already reference."""
    used = slots_used(ops, k) | set(taken)
    for j in range(NCPR):
        if j not in used:
            return j
    raise ValueError(f"operator '{ops.names[k]}' uses all {NCPR} payload slots")


def homogenize(ops, absorb_s=None):
    """Rewrite every operator so all its coefficients ride payload slots.

    Literal (cix = -1) coefficients get a per-operator GAIN slot. Operators
    named in `absorb_s` additionally get a SLOT_CONST entry on a TARGET slot,
    so a per-row `s` column can move into the payload where it folds like
    everything else.

    Returns (new_ops, plan) with plan[k] = {"gain": j|None, "target": j|None}.
    The operator's kernel HINT is dropped wherever the structure changed: a
    hand-written fused kernel has no gain slot and would silently ignore it.
    """
    absorb = set(absorb_s or ())
    out = OperatorTable(n_channels=ops.n_channels)
    plan = {}
    for k in range(ops.n_ops):
        need_gain = not is_homogeneous(ops, k)
        want_tgt = ops.names[k] in absorb
        g = free_slot(ops, k) if need_gain else None
        t = free_slot(ops, k, taken=() if g is None else (g,)) if want_tgt else None
        lin, quad, const = [], [], []
        for (slot, ch, val, cix) in ops.lin[k]:
            cix = g if cix < 0 else cix
            (const if slot == SLOT_CONST else lin).append(
                (val, cix) if slot == SLOT_CONST else (slot, ch, val, cix))
        for (s1, c1, s2, c2, val, cix) in ops.quad[k]:
            quad.append((s1, c1, s2, c2, val, g if cix < 0 else cix))
        if t is not None:
            const.append((-1.0, t))
        changed = need_gain or want_tgt
        out.add_op(ops.names[k], lin, quad,
                   kernel="" if changed else ops.kernels[k], const=const)
        plan[k] = {"gain": g, "target": t}
    return out, plan


def fold(rows, ops, plan, out_dtype=np.float32):
    """(x, op, w, s, c, m, fuzz) -> (x, op, payload, m, fuzz): the weight and
    target ride the payload, the modulus scales with them, w becomes 1 and s 0.

    `ops`/`plan` must come from `homogenize`. Refuses rather than silently
    dropping a term: a row carrying s for an operator with no target slot, or
    a non-homogeneous operator, has no folded representative."""
    out = rows.copy()
    c = out["c"].astype(np.float64)
    for k in np.unique(rows["op"]):
        m = rows["op"] == k
        p = plan[int(k)]
        if p["gain"] is not None:
            c[m, p["gain"]] = 1.0
        if p["target"] is not None:
            c[m, p["target"]] = rows["s"][m]
        elif np.any(rows["s"][m]):
            raise ValueError(
                f"operator '{ops.names[int(k)]}' has rows with s != 0 and no "
                "target slot; homogenize(absorb_s=[...]) it first")
        if not is_homogeneous(ops, int(k)):
            raise ValueError(f"operator '{ops.names[int(k)]}' is not homogeneous")
    sw = np.sqrt(np.maximum(rows["w"].astype(np.float64), 0.0))
    c *= sw[:, None]
    out["c"] = c.astype(out_dtype)
    out["w"] = 1.0
    out["s"] = 0.0
    if "nyquist" in (rows.dtype.names or ()):
        out["nyquist"] = (rows["nyquist"].astype(np.float64) * sw).astype(out["nyquist"].dtype)
    return out


def reweight(rows, factor, mask=None):
    """Multiply the effective weight of selected rows by `factor`, in place of
    the `w` column that no longer exists. Weight enters as the payload SQUARED,
    so the payload — and the modulus of a wrapped row — scale by √factor.
    Works on folded rows, which is the point: IRLS, λ knobs and stratified
    reweighting stay one-liners after folding."""
    out = rows.copy()
    m = slice(None) if mask is None else mask
    f = np.sqrt(np.maximum(np.asarray(factor, np.float64), 0.0))
    fc = f[:, None] if np.ndim(f) else f
    out["c"][m] = (out["c"][m].astype(np.float64) * fc).astype(rows["c"].dtype)
    if "nyquist" in (rows.dtype.names or ()):
        out["nyquist"][m] = (out["nyquist"][m].astype(np.float64) * f).astype(rows["nyquist"].dtype)
    return out


def gain_of(rows, ops, plan):
    """Recovered per-row weight where the operator HAS a dedicated gain slot
    (w = c[gain]²), NaN where it does not — the payload there is inseparable
    from the covector and the weight genuinely is not a property of the row
    any more. Diagnostics only; nothing in the objective needs this."""
    w = np.full(len(rows), np.nan)
    for k in np.unique(rows["op"]):
        g = plan[int(k)]["gain"]
        if g is not None:
            m = rows["op"] == k
            w[m] = rows["c"][m, g].astype(np.float64) ** 2
    return w
