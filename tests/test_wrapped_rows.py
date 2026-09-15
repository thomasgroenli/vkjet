"""Wrapped-Gaussian rows: the contract (x, op, payload, m) with r ≡ 0 (mod m).

  1. numpy: soft_unwrap is the derivative of wrapped_half_sq (finite
     differences), both are m-periodic, m = 0 is the identity, and the
     tau -> 0 limit is the hard sawtooth ½·wrap(r)².
  2. GPU: generic referee AND per-op JIT kernel reproduce the numpy oracle on
     rows whose residual spans several wraps, loss and gradient (FD of the
     GPU loss along a random direction vs the GPU gradient).
  3. m = 0 rows: the wrapped kernels reproduce the plain LS numbers exactly.
  4. per-op JIT vs generic parity on mixed-m rows (verify_generated).

Run: PYTHONPATH=~/projects/volkano:~/genspline .venv/bin/python tests/test_wrapped_rows.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vkjet.context import Context                                  # noqa: E402
from vkjet.data import Axes, data_operator                         # noqa: E402
from vkjet.eqrow import (EqRowTerm, NCH, NCPR, soft_unwrap,        # noqa: E402
                          wrapped_half_sq, gather_fields_jet,
                          row_residuals_oracle)
from vkjet import genkernel as gk                                  # noqa: E402

LO, HI, GRID = (0., 0., 0., 0.), (1., 6., 6., 10.), (4, 4, 4, 6)


def test_numpy_oracle():
    rng = np.random.default_rng(0)
    r = rng.uniform(-3, 3, 4000)
    for tau in (0.0, 0.1, 0.3, 0.5, -1.0):
        m = 1.0
        h = 1e-6
        fd = (wrapped_half_sq(r + h, m, tau) - wrapped_half_sq(r - h, m, tau)) / (2 * h)
        g = soft_unwrap(r, m, tau)
        err = np.abs(fd - g).max()
        assert err < 2e-5, (tau, err)
        # periodicity of both
        assert np.allclose(soft_unwrap(r + 2 * m, m, tau), g, atol=1e-9)
        assert np.allclose(wrapped_half_sq(r - 3 * m, m, tau),
                           wrapped_half_sq(r, m, tau), atol=1e-9)
    # tau < 0: the pure cosine loss, first harmonic only
    k = 2 * np.pi
    assert np.allclose(soft_unwrap(r, 1.0, -1.0), np.sin(k * r) / k)
    # m = 0 identity
    assert np.array_equal(soft_unwrap(r, 0.0, 0.3), r)
    assert np.array_equal(wrapped_half_sq(r, 0.0, 0.3), 0.5 * r * r)
    # tau -> 0: hard sawtooth (away from the half-wrap)
    m = 0.7
    wr = r - m * np.round(r / m)
    ok = np.abs(np.abs(wr) - m / 2) > 0.05
    assert np.allclose(soft_unwrap(r, m, 0.0)[ok], wr[ok], atol=1e-6)
    assert np.allclose(wrapped_half_sq(r, m, 0.0)[ok], 0.5 * wr[ok] ** 2, atol=1e-6)
    # r~ vanishes at the half-wrap (the two dominant branches cancel; the
    # omitted third shell leaves an e^(-1/tau²)-sized remainder) and is odd
    assert abs(soft_unwrap(np.array([m / 2]), m, 0.3)[0]) < 1e-4 * m
    assert np.allclose(soft_unwrap(-r, m, 0.3), -soft_unwrap(r, m, 0.3))
    print("  numpy oracle: FD, periodicity, m=0 identity, tau->0 limit  OK")


def _setup(ctx):
    axes = Axes(LO, HI, GRID, periodic=(True, False, False, False))
    bases = axes.bases()
    iw = [GRID[k] / (HI[k] - LO[k]) for k in range(4)]
    nco = int(np.prod([b.primal_extent for b in bases])) * NCH
    return axes, bases, iw, nco


def _rows(bases, rng, n=96):
    x = gk.verify_points(bases, rng, n)
    ops, oid = data_operator()
    c = np.zeros((n, NCPR), np.float32)
    c[:, 2] = 1.0                                 # r = w - s (unit z covector)
    w = rng.uniform(0.5, 1.5, n).astype(np.float32)
    s = rng.uniform(-1.5, 1.5, n).astype(np.float32)   # spans several wraps
    return x, ops, np.zeros(n, np.int32), w, s, c


def _gpu_loss_grad(ctx, term, cb, nco):
    lb = ctx.buffer(4); gb = ctx.buffer(nco * 4)
    lb.zero(); term.loss(cb, lb)
    gb.zero(); term.accumulate(cb, gb)
    return float(lb.download(np.float32, 1)[0]), gb.download(np.float32, nco)


def test_gpu_vs_oracle(ctx):
    axes, bases, iw, nco = _setup(ctx)
    rng = np.random.default_rng(3)
    x, ops, op, w, s, c = _rows(bases, rng)
    n = len(x)
    C = (rng.standard_normal(nco) * 0.6).astype(np.float32)
    mod = np.where(rng.random(n) < 0.7, 0.5, 0.0).astype(np.float32)   # m = 2·venc
    for tau in (0.25, -1.0):
        _gpu_vs_oracle_at(ctx, bases, iw, nco, rng, x, ops, op, w, s, c, mod,
                          tau, C)


def _gpu_vs_oracle_at(ctx, bases, iw, nco, rng, x, ops, op, w, s, c, mod, tau, C):
    cb = ctx.buffer(nco * 4); cb.upload(C)
    fields = gather_fields_jet(bases, iw, x, C.astype(np.float64).reshape(-1, NCH)
                               .reshape([b.primal_extent for b in bases] + [NCH]))
    r = row_residuals_oracle(fields, ops, op, w, s, c)
    assert (np.abs(r[mod > 0]) > mod[mod > 0]).any(), "test rows must span > 1 wrap"
    L_ref = float(np.sum(w * wrapped_half_sq(r, mod, tau)))

    gen = EqRowTerm(ctx, bases, iw, ops)
    gen.tau = tau
    gen.bind_batch(x, op, w, s, c, modulus=mod)
    jit = gk.GeneratedRowTerm(ctx, bases, iw, ops, 0)
    jit.tau = tau
    jit.bind_batch(x, op, w, s, c, modulus=mod)
    for name, t in (("generic", gen), ("jit", jit)):
        L, g = _gpu_loss_grad(ctx, t, cb, nco)
        assert abs(L - L_ref) < 2e-4 * abs(L_ref), (name, L, L_ref)
        # gradient: FD of the GPU loss along a random direction
        v = rng.standard_normal(nco).astype(np.float32); v /= np.linalg.norm(v)
        h = 2e-3
        cp = ctx.buffer(nco * 4); cp.upload(C + h * v)
        cm = ctx.buffer(nco * 4); cm.upload(C - h * v)
        Lp = _gpu_loss_grad(ctx, t, cp, nco)[0]
        Lm = _gpu_loss_grad(ctx, t, cm, nco)[0]
        fd = (Lp - Lm) / (2 * h)
        gv = float(g.astype(np.float64) @ v)
        assert abs(fd - gv) < 2e-2 * max(abs(fd), 1e-3), (name, fd, gv)
        print(f"  {name:8s} tau={tau:+.2f}: loss {L:.6f} vs oracle {L_ref:.6f}; "
              f"grad·v {gv:+.5f} vs FD {fd:+.5f}  OK")


def test_m0_identity(ctx):
    """With m = 0 everywhere the wrapped kernels equal the plain LS kernels,
    for any tau: same loss, same gradient (bit-for-bit up to atomic order)."""
    axes, bases, iw, nco = _setup(ctx)
    rng = np.random.default_rng(5)
    x, ops, op, w, s, c = _rows(bases, rng)
    C = (rng.standard_normal(nco) * 0.6).astype(np.float32)
    cb = ctx.buffer(nco * 4); cb.upload(C)
    fields = gather_fields_jet(bases, iw, x, C.astype(np.float64)
                               .reshape([b.primal_extent for b in bases] + [NCH]))
    r = row_residuals_oracle(fields, ops, op, w, s, c)
    L_ref = float(0.5 * np.sum(w * r * r))
    for name, mk in (("generic", lambda: EqRowTerm(ctx, bases, iw, ops)),
                     ("jit", lambda: gk.GeneratedRowTerm(ctx, bases, iw, ops, 0))):
        a = mk(); a.bind_batch(x, op, w, s, c)                       # no modulus
        b = mk(); b.tau = 0.4; b.bind_batch(x, op, w, s, c, modulus=np.zeros(len(x), np.float32))
        La, ga = _gpu_loss_grad(ctx, a, cb, nco)
        Lb, gb = _gpu_loss_grad(ctx, b, cb, nco)
        assert abs(La - L_ref) < 1e-4 * abs(L_ref), (name, La, L_ref)
        assert abs(La - Lb) < 1e-6 * abs(La), (name, La, Lb)
        assert np.linalg.norm(ga - gb) < 1e-5 * np.linalg.norm(ga), name
        print(f"  {name:8s}: m=0 identity  loss {La:.6f} == {Lb:.6f}  OK")


def test_parity_mixed_m(ctx):
    axes, bases, iw, nco = _setup(ctx)
    ops, _ = data_operator()
    t = gk.GeneratedRowTerm(ctx, bases, iw, ops, 0)
    gk.verify_generated(ctx, t, bases, iw, ops, 0, force=True)
    print("  verify_generated (mixed m, tau=0.3): referee vs JIT parity  OK")


if __name__ == "__main__":
    test_numpy_oracle()
    ctx = Context()
    print(f"device: {ctx.device_name}")
    test_gpu_vs_oracle(ctx)
    test_m0_identity(ctx)
    test_parity_mixed_m(ctx)
    ctx.destroy()
    print("ALL OK")
