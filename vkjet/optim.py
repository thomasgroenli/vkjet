"""Matrix-free Gauss-Newton CG with Levenberg-Marquardt damping.

The production optimiser: H_GN.v is supplied by the terms themselves via
``.hvp(coef, v, out)``, so the normal equations are never formed. Per step

    g   = sum_t t.accumulate(coef, grad)          gradient
    D   = sum_t t.accumulate_diag(coef, diag)     diag(H_GN), Jacobi preconditioner
    dx  = PCG solve of (H_GN + mu*D) dx = -g      cg_iters matvecs via t.hvp
    accept/reject on the TRUE loss from t.loss, mu adapted from the gain ratio

Truncating CG at a fixed ``cg_iters`` is deliberate Krylov regularisation, not
a convergence compromise. All persistent vectors (coef, grad, diag, p, Ap, ...)
are allocated once per solver and reused; CG scalars stay device-resident and
the whole solve is captured into one command submission.

A "term" is any object exposing accumulate / accumulate_diag / hvp / loss over
a shared coefficient buffer - in vkjet that is EqRowTerm and the JIT-generated
kernels from :mod:`vkjet.genkernel`.
"""
from __future__ import annotations

import math
import os
import struct

import numpy as np

from .context import Context, STORAGE

SHADER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "shaders", "spv")
PRECOND_SPV = os.path.join(SHADER_DIR, "precond_step.spv")
PRECOND_TRIAL_SPV = os.path.join(SHADER_DIR, "precond_trial.spv")
VEC_DOT_SPV = os.path.join(SHADER_DIR, "vec_dot.spv")
VEC_AXPBY_SPV = os.path.join(SHADER_DIR, "vec_axpby.spv")
VEC_PDIV_SPV = os.path.join(SHADER_DIR, "vec_pdiv.spv")
VEC_MADD_DIAG_SPV = os.path.join(SHADER_DIR, "vec_madd_diag.spv")
VEC_MAX_SPV = os.path.join(SHADER_DIR, "vec_max.spv")
VEC_AXPBY_S_SPV = os.path.join(SHADER_DIR, "vec_axpby_s.spv")
CG_ALPHA_SPV = os.path.join(SHADER_DIR, "cg_alpha.spv")
CG_BETA_SPV = os.path.join(SHADER_DIR, "cg_beta.spv")


class _MaxReduce:
    """GPU max-reduction of a (non-negative) buffer → host scalar. diag(H_GN) ≥ 0, so the
    float bit-pattern is monotonic: atomicMax on uint(bits) yields the max without
    downloading the whole buffer. Replaces ``diag.download(...).max()`` (a full ~10 MB
    per-step round-trip at the fine stage) with one kernel + a 4-byte read."""

    def __init__(self, ctx: Context, n: int):
        self.ctx = ctx; self.n = int(n)
        self.umax = ctx.buffer(4)
        self.nmeta = ctx.buffer(4, device_local=False)
        self.nmeta.upload(struct.pack("<i", self.n))
        self.prog = ctx.program(VEC_MAX_SPV, bindings=[STORAGE] * 3)

    def __call__(self, buf):
        self.umax.zero()
        self.ctx.run(self.prog, [buf, self.umax, self.nmeta],
                     groups=min((self.n + 255) // 256, 4096))
        return float(self.umax.download(np.float32, 1)[0]) + 1e-30   # uint bits == float bits


class NormTestBatcher:
    """Self-calibrating gradient accumulation (Byrd–Bollapragada norm test).

    Sums fixed-size minibatches (the hardware work-unit) into a gradient estimate and
    stops when the across-minibatch variance falls below tol²·‖ĝ‖² — i.e. when the
    estimate is statistically stable. The effective batch tracks the gradient SNR, NOT
    the dataset size, so it self-adapts to unknown over/under-sampling: massively
    oversampled → a few minibatches; scarce/near-optimum → (nearly) all of them.
    """

    def __init__(self, ctx: Context, n_params: int, coord_var: bool = False):
        self.ctx = ctx
        self.n = int(n_params)
        self.gk = ctx.buffer(self.n * 4)
        self.M = ctx.buffer(self.n * 4)         # accumulated data minibatch sum
        self.combined = ctx.buffer(self.n * 4)  # scale·M + G_det (combined-norm scratch)
        self.scalar = ctx.buffer(4)
        self.vmeta = ctx.buffer(12, device_local=False)
        self.dot_p = ctx.program(VEC_DOT_SPV, bindings=[STORAGE] * 4)
        self.axpby_p = ctx.program(VEC_AXPBY_SPV, bindings=[STORAGE] * 4)
        # per-COORDINATE variance of the data-gradient estimate (SNR gate consumer):
        # alongside M = Σ_k g_k also accumulate Q = Σ_k g_k∘g_k elementwise.
        self.coordv = bool(coord_var)
        if self.coordv:
            self.Q = ctx.buffer(self.n * 4)
            self.madd_p = ctx.program(VEC_MADD_DIAG_SPV, bindings=[STORAGE] * 4)
        self._K_used = 0; self._nmb = 0
        # SNR state recorded by gradient_combined (see its docstring)
        self.last_var = None; self.last_norm2 = None
        self.passed = False; self.exhausted = False

    def _g(self): return min((self.n + 255) // 256, 4096)
    def _vm(self, a, b): self.vmeta.upload(struct.pack("<i2f", self.n, a, b)); return self.vmeta

    def _dot(self, a, b):
        self.scalar.zero()
        self.ctx.run(self.dot_p, [a, b, self.scalar, self._vm(0.0, 0.0)], groups=self._g())
        return float(self.scalar.download(np.float32, 1)[0])

    def _axpby(self, x, y, a, b, z):
        self.ctx.run(self.axpby_p, [x, y, z, self._vm(a, b)], groups=self._g())

    def gradient(self, term, coef_buf, grad_out, tol=0.1, order=None):
        """Accumulate minibatches of `term` (a FusedDataGrad bound via bind_minibatches)
        into grad_out, scaled to the full population. Returns (used_minibatch_indices, scale)
        — the SAME set must drive the HVP so CG sees a consistent operator.
        """
        mbs = term.minibatches
        nmb = len(mbs)
        order = list(order) if order is not None else list(range(nmb))
        grad_out.zero()
        Q = 0.0; used = []
        for k in order:
            self.gk.zero()
            term.accumulate(coef_buf, self.gk, batch=mbs[k])      # minibatch gradient (sum)
            Q += self._dot(self.gk, self.gk)
            self._axpby(self.gk, grad_out, 1.0, 1.0, grad_out)    # M += gk
            used.append(k); K = len(used)
            if K >= 2:
                m2 = self._dot(grad_out, grad_out)
                if (Q - m2 / K) * K / (K - 1) <= tol * tol * m2:  # Var(ĝ) ≤ tol²‖ĝ‖²
                    break
        scale = float(nmb) / len(used)
        if scale != 1.0:
            self._axpby(grad_out, grad_out, scale, 0.0, grad_out)  # scale to full population
        return used, scale

    def gradient_combined(self, term, coef_buf, grad, tol=0.15, order=None):
        """Combined-gradient norm test. ``grad`` enters holding the DETERMINISTIC baseline
        G_det (wall+pde, accumulated first), unmodified until the end. The data minibatch
        sum M accumulates separately; the stopping test compares the data-estimate variance
        to the COMBINED norm ‖scale·M + G_det‖ — so cancellation between data and PDE near
        the solution shrinks the denominator and forces a larger effective batch. Leaves
        grad = G_det + scale·M = G_combined. Returns (used_indices, scale).

        Records the SNR state for the two §3.3/§3.4 consumers: ``last_var`` (variance of the
        scaled data estimate — its √ is the CG noise-forcing target), ``last_norm2``
        (‖G_combined‖² at the last test), ``passed`` (the norm test was met), ``exhausted``
        (all minibatches consumed). exhausted ∧ ¬passed = the FULL-batch gradient fails its
        own SNR test — the statistical stopping signal ("the gradient is noise").
        """
        mbs = term.minibatches
        nmb = len(mbs)
        order = list(order) if order is not None else list(range(nmb))
        self.M.zero(); Q = 0.0; used = []
        if self.coordv:
            self.Q.zero()
        self.last_var = None; self.last_norm2 = None; self.passed = False
        for k in order:
            self.gk.zero()
            term.accumulate(coef_buf, self.gk, batch=mbs[k])
            Q += self._dot(self.gk, self.gk)
            if self.coordv:                                       # Q_i += gk_i²
                self.ctx.run(self.madd_p, [self.gk, self.gk, self.Q,
                                           self._vm(1.0, 0.0)], groups=self._g())
            self._axpby(self.gk, self.M, 1.0, 1.0, self.M)        # M += gk
            used.append(k); K = len(used)
            # tol ≤ 0 ⇒ pure variance ESTIMATION (full batch by construction):
            # the per-minibatch stopping test can never pass, so skip its dots +
            # downloads and compute the estimate once, at the final K.
            if K >= 2 and (tol > 0.0 or K == nmb):
                mM = self._dot(self.M, self.M)
                var = float(nmb * nmb) * ((Q - mM / K) / (K - 1)) / K   # Var(scaled data est.)
                self._axpby(self.M, grad, float(nmb) / K, 1.0, self.combined)  # scale·M + G_det
                self.last_var = var
                self.last_norm2 = self._dot(self.combined, self.combined)
                if tol > 0.0 and var <= tol * tol * self.last_norm2:
                    self.passed = True
                    break
        self.exhausted = len(used) == nmb
        self._K_used = len(used); self._nmb = nmb
        scale = float(nmb) / len(used)
        self._axpby(self.M, grad, scale, 1.0, grad)              # grad = G_det + scale·M
        return used, scale

    def coord_var_host(self):
        """Per-coordinate variance of the SCALED data-gradient estimate from the last
        gradient_combined call (requires coord_var=True): Var_i = nmb²·(Q_i − M_i²/K)
        / (K(K−1)) — the elementwise analog of last_var. Clipped at 0 (the fp32
        Q − M²/K cancellation only matters where signal ≫ noise, i.e. where the
        gate is inactive anyway). Host round-trip: 2 downloads of n floats."""
        K, nmb = self._K_used, self._nmb
        if not self.coordv or K < 2:
            return np.zeros(self.n)
        Q = self.Q.download(np.float32, self.n).astype(np.float64)
        M = self.M.download(np.float32, self.n).astype(np.float64)
        return np.maximum(float(nmb * nmb) * (Q - M * M / K) / (K * (K - 1)), 0.0)


class GaussNewtonCG:
    """Matrix-free Gauss-Newton with preconditioned-CG inner solve + LM outer loop.

    Each step solves the damped normal equations (H_GN + μ·D)·δ = −g by CG, where
    H_GN·v is supplied matrix-free by the terms' .hvp(coef, v, out) (the same
    gather/scatter kernels as the gradient) and D = diag(H_GN) is the Jacobi
    preconditioner. The LM damping μ self-adapts from the loss (accept+shrink /
    reject+grow), so there is NO manual step-size knob and convergence is
    Newton-fast (the CG conditions the FULL operator, off-diagonal included).

        opt = GaussNewtonCG(ctx, n); opt.set_coef(c0)
        for k: opt.step([data, wall, pde], loss_fn)   # each term: accumulate/accumulate_diag/hvp/loss

    EXACT-POLYNOMIAL extensions (the loss is an exact QUARTIC in the coefficients on
    unwrapped data: data/wall residuals linear in c, PDE residual quadratic):

    - step(..., line_search=True): exact quartic line search along the CG direction —
      5 loss kernels pin L(c+αδ) exactly, the minimizing α is closed-form. Replaces the
      LM accept/reject heuristic (a reject then costs 5 cheap loss evals, not a full
      CG re-solve; μ adapts from the accepted α instead of ρ).
    - step(..., newton_eps=ε>0): FULL-Newton HVP by central gradient differencing —
      ∇L is an exact CUBIC, so (∇L(c+εv) − ∇L(c−εv))/2ε has ZERO truncation error at
      any ε (only fp32 rounding; ε is auto-scaled to ‖c‖/‖v‖). Terms with the
      `newton_fd` attribute (nonlinear residual: FusedPDEGrad/ALM) use it; linear terms
      keep their GN hvp, which is already their exact Hessian. CG gains a Steihaug
      negative-curvature guard. Matters when the residual is NOT small at the solution
      (the PDE physics floor) — exactly where GN ≠ Newton systematically.
    """

    def __init__(self, ctx: Context, n_params: int):
        self.ctx = ctx
        self.n = int(n_params)
        self.diag_max = 1.0
        self.mu_rel = 1e-2      # CG conditions the off-diagonal → μ only covers the null space
        self.nu = 2.0           # Nielsen reject-growth factor
        self.last_loss = None
        self.last_pred = None   # predicted decrease of the last ACCEPTED step: the damped,
        self.last_actual = None # Krylov-truncated Newton decrement (affine-invariant, units
                                # of the objective) and the measured decrease beside it
        self.last_alpha = 1.0   # accepted line-search step (diagnostic)
        self.gate_frac = 0.0    # fraction of coordinates SNR-gated (diagnostic)
        self._cnorm = 0.0
        self._xt = None; self._gt = None   # Newton central-difference scratch (lazy)
        mk = lambda: ctx.buffer(self.n * 4)
        self.coef = mk()
        self.grad = mk(); self.grad.zero()
        self.diag = mk(); self.diag.zero()
        self.backup = mk()
        self.delta = mk(); self.r = mk(); self.z = mk(); self.p = mk(); self.Ap = mk()
        self.scalar = ctx.buffer(4)
        self.vmeta = ctx.buffer(12, device_local=False)
        self.dot_p = ctx.program(VEC_DOT_SPV, bindings=[STORAGE] * 4)
        self.axpby_p = ctx.program(VEC_AXPBY_SPV, bindings=[STORAGE] * 4)
        self.pdiv_p = ctx.program(VEC_PDIV_SPV, bindings=[STORAGE] * 4)
        self.madd_p = ctx.program(VEC_MADD_DIAG_SPV, bindings=[STORAGE] * 4)
        self._maxred = _MaxReduce(ctx, self.n)
        self.bpx = None         # optional multilevel preconditioner (set_preconditioner)
        # -- device-resident CG scalars: α/β never round-trip to the host -------- #
        self.nmeta = ctx.buffer(4, device_local=False); self.nmeta.upload(struct.pack("<i", self.n))
        self.s_rz = ctx.buffer(4); self.s_pAp = ctx.buffer(4); self.s_rzn = ctx.buffer(4)
        self.s_alpha = ctx.buffer(4); self.s_nalpha = ctx.buffer(4); self.s_beta = ctx.buffer(4)
        self.s_one = ctx.buffer(4); self.s_one.upload(struct.pack("<f", 1.0))
        self.axpby_s_p = ctx.program(VEC_AXPBY_S_SPV, bindings=[STORAGE] * 6)
        self.cg_alpha_p = ctx.program(CG_ALPHA_SPV, bindings=[STORAGE] * 4)
        self.cg_beta_p = ctx.program(CG_BETA_SPV, bindings=[STORAGE] * 3)
        # -- batched CG (one submit for the whole inner solve) ---------------- #
        # Captured dispatches read metas at EXECUTION time, so every captured
        # host constant gets its own buffer: μ is the only per-solve value.
        self._seq = None; self._seq_key = None
        mkvm = lambda a, b: self._static_vm(a, b)
        self.vm_m10 = mkvm(-1.0, 0.0)     # r = −g
        self.vm_00 = mkvm(0.0, 0.0)       # dot products
        self.vm_mu = ctx.buffer(12, device_local=False)   # μ, re-uploaded per solve

    def _static_vm(self, a, b):
        buf = self.ctx.buffer(12, device_local=False)
        buf.upload(struct.pack("<i2f", self.n, a, b))
        return buf

    def set_coef(self, arr): self.coef.upload(np.ascontiguousarray(arr, np.float32).reshape(-1))
    def get_coef(self): return self.coef.download(np.float32, self.n)

    # -- GPU vector primitives ------------------------------------------------ #
    def _g(self): return min((self.n + 255) // 256, 4096)
    def _vm(self, a, b): self.vmeta.upload(struct.pack("<i2f", self.n, a, b)); return self.vmeta

    def _dot(self, a, b):
        self.scalar.zero()
        self.ctx.run(self.dot_p, [a, b, self.scalar, self._vm(0.0, 0.0)], groups=self._g())
        return float(self.scalar.download(np.float32, 1)[0])

    def _dot_to(self, a, b, out):      # out[0] = Σ a·b, stays on device (no download)
        out.zero()
        self.ctx.run(self.dot_p, [a, b, out, self._vm(0.0, 0.0)], groups=self._g())

    def _axpby(self, x, y, a, b, z):   # z = a·x + b·y
        self.ctx.run(self.axpby_p, [x, y, z, self._vm(a, b)], groups=self._g())

    def _axpby_s(self, x, y, a_buf, b_buf, z):   # z = a_buf[0]·x + b_buf[0]·y (device scalars)
        self.ctx.run(self.axpby_s_p, [x, y, z, a_buf, b_buf, self.nmeta], groups=self._g())

    def _pdiv(self, x, d, a, z):       # z = x / (d + a)
        self.ctx.run(self.pdiv_p, [x, d, z, self._vm(a, 0.0)], groups=self._g())

    def _madd_diag(self, x, d, a, y):  # y += a · d · x
        self.ctx.run(self.madd_p, [x, d, y, self._vm(a, 0.0)], groups=self._g())

    def set_preconditioner(self, bpx):
        """Plug in a multilevel (BPX) preconditioner (any object exposing apply(r, z)) to replace
        the nodal-diagonal CG preconditioner — gives scale-uniform conditioning so a cold solve
        at the finest grid matches the staged ladder. None ⇒ nodal diagonal (default)."""
        self.bpx = bpx

    def _precond(self, r, z, mu_abs):  # z = M⁻¹ r
        if self.bpx is not None:
            self.bpx.apply(r, z)                               # multilevel (fixed-floor) BPX
        else:
            self._pdiv(r, self.diag, mu_abs, z)                # nodal damped-diagonal Jacobi

    # -- exact full-Hessian action (polynomial structure) --------------------- #
    def _hvp_full(self, terms, p, out, eps_rel):
        """out += H_full·p. Terms tagged `newton_fd` (quadratic residual ⇒ CUBIC gradient)
        get the exact central difference (∇L(c+εp) − ∇L(c−εp))/2ε — zero truncation error
        for any ε, so ε is chosen purely against fp32 rounding (a fixed fraction of the
        coefficient scale). Linear-residual terms keep their GN hvp (= exact Hessian)."""
        fd = [t for t in terms if getattr(t, "newton_fd", False)]
        for t in terms:
            if t not in fd:
                t.hvp(self.coef, p, out)
        if not fd:
            return
        if self._xt is None:
            self._xt = self.ctx.buffer(self.n * 4)
            self._gt = self.ctx.buffer(self.n * 4)
        pn = math.sqrt(max(self._dot(p, p), 1e-30))
        eps = eps_rel * (self._cnorm + 1.0) / pn
        for sgn in (1.0, -1.0):
            self._axpby(p, self.coef, sgn * eps, 1.0, self._xt)    # x± = c ± ε·p
            self._gt.zero()
            for t in fd:
                acc = getattr(t, "accumulate_replay", None) or t.accumulate
                acc(self._xt, self._gt)
            self._axpby(self._gt, out, sgn / (2.0 * eps), 1.0, out)

    # -- preconditioned CG: solve (H + μI_D)·δ = −g --------------------------- #
    def _pcg(self, terms, mu_abs, cg_iters, cg_tol, cg_noise=0.0, newton_eps=0.0):
        """GPU-resident PCG: rz/pAp/α/β live in 1-float device buffers (cg_alpha/cg_beta
        write α/β; axpby_s reads them) so no CG scalar round-trips to the host.

        cg_noise=0 (default): FIXED iteration count — truncating at cg_iters IS the
        Krylov regularization, no early-stop download (bit-identical legacy behavior).
        cg_noise>0: solve-to-gradient-noise forcing (design §3.3) — stop as soon as
        ‖r_cg‖ ≤ cg_noise (θ·√Var(ĝ): solving the Newton system more accurately than
        the gradient's own statistical error is wasted work). Costs one 4-byte ‖r‖
        download per iteration; regularization is then carried by λ/μ, not truncation."""
        self.delta.zero()
        self._axpby(self.grad, self.grad, -1.0, 0.0, self.r)   # r = −grad   (setup, host const)
        self._precond(self.r, self.z, mu_abs)                  # z = M⁻¹r
        self.ctx.copy_buffer(self.z, self.p, self.n * 4)
        self._dot_to(self.r, self.z, self.s_rz)                # rz on device
        for used in range(1, cg_iters + 1):
            self.Ap.zero()
            if newton_eps > 0.0:
                self._hvp_full(terms, self.p, self.Ap, newton_eps)  # Ap += H_full·p
            else:
                for t in terms:
                    t.hvp(self.coef, self.p, self.Ap)          # Ap += H_GN·p
            self._madd_diag(self.p, self.diag, mu_abs, self.Ap)  # Ap += μ·D·p
            self._dot_to(self.p, self.Ap, self.s_pAp)          # pAp on device
            if newton_eps > 0.0:
                # H_full may be indefinite — Steihaug guard (4-byte read per iter):
                # stop at the last conjugate point; on first-iter negative curvature take
                # the preconditioned-gradient direction (the line search sets its length).
                if float(self.s_pAp.download(np.float32, 1)[0]) <= 0.0:
                    if used == 1:
                        self.ctx.copy_buffer(self.p, self.delta, self.n * 4)
                    return used
            self.ctx.run(self.cg_alpha_p, [self.s_rz, self.s_pAp, self.s_alpha, self.s_nalpha],
                         groups=1)                             # α = rz/pAp, −α  (on device)
            self._axpby_s(self.p, self.delta, self.s_alpha, self.s_one, self.delta)  # δ += α·p
            self._axpby_s(self.Ap, self.r, self.s_nalpha, self.s_one, self.r)        # r −= α·Ap
            if cg_noise > 0.0 and self._dot(self.r, self.r) <= cg_noise * cg_noise:
                return used                                    # solved to the gradient noise
            self._precond(self.r, self.z, mu_abs)
            self._dot_to(self.r, self.z, self.s_rzn)           # rz_new on device
            self.ctx.run(self.cg_beta_p, [self.s_rzn, self.s_rz, self.s_beta], groups=1)  # β; rz←rz_new
            self._axpby_s(self.z, self.p, self.s_one, self.s_beta, self.p)           # p = z + β·p
        return cg_iters

    # -- exact quartic line search along δ ------------------------------------ #
    def _quartic_ls(self, loss_fn, L0, alphas=(0.25, 0.5, 1.0, 2.0), alpha_cap=8.0):
        """L(backup + α·δ) is an EXACT quartic in α (data/wall linear, PDE quadratic in c;
        the α⁴ coefficient Σλ‖B(δ,δ)‖² ≥ 0, so it is bounded below). Four loss kernels
        (plus the known L0) pin the polynomial exactly; the minimizer of its cubic
        derivative is closed-form. One final evaluation at α* + argmin over ALL evaluated
        points keeps the step optimal-within-fp32-noise even when exactness is broken
        (wrap-aware cosine data loss). Leaves coef = backup + α_best·δ; returns
        (α_best, L(α_best)) — α_best = 0 means no descent anywhere on the ray."""
        evals = {0.0: float(L0)}
        for a in alphas:
            self._axpby(self.delta, self.backup, a, 1.0, self.coef)
            evals[a] = float(loss_fn())
        y = np.array([evals[a] - L0 for a in alphas], np.float64)
        astar = None
        if np.isfinite(y).all():
            A = np.array([[a ** k for k in range(1, 5)] for a in alphas], np.float64)
            try:
                c1, c2, c3, c4 = np.linalg.solve(A, y)
                roots = np.roots([4.0 * c4, 3.0 * c3, 2.0 * c2, c1])
                cand = [float(r.real) for r in roots
                        if abs(r.imag) < 1e-9 * (1.0 + abs(r)) and 1e-6 < r.real <= alpha_cap]
                if cand:
                    q = lambda a: (((c4 * a + c3) * a + c2) * a + c1) * a
                    astar = min(cand, key=q)
            except np.linalg.LinAlgError:
                pass
        if astar is not None and min(abs(astar - a) for a in evals) > 1e-3:
            self._axpby(self.delta, self.backup, astar, 1.0, self.coef)
            evals[astar] = float(loss_fn())
        fin = {a: v for a, v in evals.items() if np.isfinite(v)}
        a_best = min(fin, key=fin.get)
        self._axpby(self.delta, self.backup, a_best, 1.0, self.coef)  # α=0 ⇒ restores backup
        return a_best, fin[a_best]

    # -- batched PCG: the whole fixed-iteration solve in ONE submit ----------- #
    def _pcg_batched(self, terms, mu_abs, cg_iters):
        """Bit-for-bit the `_pcg` dispatch chain (fixed iterations, diagonal or
        BPX preconditioner), recorded once into a CommandSequence and re-submitted
        every solve — one fence instead of ~12·cg_iters. μ is the only per-solve
        host constant; it lives in its own meta buffer, re-uploaded before each
        submit (an LM retry with a new μ therefore needs NO re-record). The
        sequence is invalidated when the term set, their bound batches or the
        preconditioner change (a new stage)."""
        self.vm_mu.upload(struct.pack("<i2f", self.n, mu_abs, 0.0))
        key = (cg_iters, id(self.bpx),
               tuple((id(t), id(getattr(t, "_batch", None))) for t in terms))

        def precond(r, z):                   # z = M^-1 r, capturable either way
            if self.bpx is not None:
                self.bpx.apply(r, z)         # per-level metas are static
            else:
                run(self.pdiv_p, [r, self.diag, z, self.vm_mu], groups=g)
        if self._seq_key != key:
            if self._seq is None:
                self._seq = self.ctx.sequence()
            else:
                self._seq.reset()
            g = self._g()
            run = self.ctx.run
            with self.ctx.capture(self._seq):
                self.delta.zero()
                run(self.axpby_p, [self.grad, self.grad, self.r, self.vm_m10],
                    groups=g)                                       # r = −g
                precond(self.r, self.z)
                self.ctx.copy_buffer(self.z, self.p, self.n * 4)
                self.s_rz.zero()
                run(self.dot_p, [self.r, self.z, self.s_rz, self.vm_00], groups=g)
                for _ in range(cg_iters):
                    self.Ap.zero()
                    for t in terms:
                        t.hvp(self.coef, self.p, self.Ap)           # Ap += H·p
                    run(self.madd_p, [self.p, self.diag, self.Ap, self.vm_mu],
                        groups=g)                                   # Ap += μ·D·p
                    self.s_pAp.zero()
                    run(self.dot_p, [self.p, self.Ap, self.s_pAp, self.vm_00],
                        groups=g)
                    run(self.cg_alpha_p,
                        [self.s_rz, self.s_pAp, self.s_alpha, self.s_nalpha],
                        groups=1)
                    run(self.axpby_s_p, [self.p, self.delta, self.delta,
                                         self.s_alpha, self.s_one, self.nmeta],
                        groups=g)                                   # δ += α·p
                    run(self.axpby_s_p, [self.Ap, self.r, self.r,
                                         self.s_nalpha, self.s_one, self.nmeta],
                        groups=g)                                   # r −= α·Ap
                    precond(self.r, self.z)                         # z = M⁻¹r
                    self.s_rzn.zero()
                    run(self.dot_p, [self.r, self.z, self.s_rzn, self.vm_00],
                        groups=g)
                    run(self.cg_beta_p, [self.s_rzn, self.s_rz, self.s_beta],
                        groups=1)                                   # β; rz ← rz_new
                    run(self.axpby_s_p, [self.z, self.p, self.p,
                                         self.s_one, self.s_beta, self.nmeta],
                        groups=g)                                   # p = z + β·p
            self._seq.record()
            self._seq_key = key
        self._seq.submit()
        return cg_iters

    def step(self, terms, loss_fn, cg_iters: int = 25, cg_tol: float = 1e-3,
             beta: float = 3.0, mu_rel_min: float = 1e-6, mu_rel_max: float = 1e3,
             max_tries: int = 4, cg_noise=0.0, line_search: bool = False,
             newton_eps: float = 0.0, snr_gate=None, snr_theta: float = 2.0,
             snr_cap: float = 1e4):
        """One GN-CG step with LM μ-adaptation. Returns (loss, accepted, cg_iters_used).

        cg_noise: absolute CG residual target (design §3.3 noise forcing) — a float, or a
        callable resolved AFTER the gradient accumulation (so a MinibatchedTerm can supply
        θ·√Var(ĝ) from the norm test it just ran). 0 ⇒ fixed cg_iters (legacy).
        line_search: exact quartic line search along δ instead of LM accept/reject
        (see class docstring). newton_eps > 0: full-Newton HVP in CG (ditto).

        snr_gate: per-COORDINATE SNR gate — the deterministic second-order analog of
        SNR-gated-minibatch Adam. A callable (resolved after gradient accumulation)
        returning Var_i of the data-gradient estimate (NormTestBatcher(coord_var=True)
        .coord_var_host). The damping diagonal is boosted per coordinate,
        D_i ← D_i·(1 + θ²·Var_i/g_i²) capped at snr_cap: coordinates whose TOTAL
        gradient (incl. the deterministic wall/PDE part) is below the data-noise floor
        are frozen — the data cannot honestly move them and physics has nothing to say —
        while coordinates with real signal (from either source) step at Newton speed.
        Since the data gradient vanishes identically along z-null directions, this
        implements a directionally adaptive data/physics weighting: data governs where
        its per-coordinate SNR is high, physics elsewhere. Records the gated fraction
        in self.gate_frac."""
        nbytes = self.n * 4
        self.last_pred = self.last_actual = None
        self.grad.zero()
        for t in terms:
            t.accumulate(self.coef, self.grad)                 # g = ∇L
        noise = float(cg_noise() if callable(cg_noise) else cg_noise)
        self.diag.zero()
        for t in terms:
            t.accumulate_diag(self.coef, self.diag)            # D = diag(H_GN)
        self.diag_max = self._maxred(self.diag)            # GPU max-reduction (no full download)
        if snr_gate is not None:
            V = np.maximum(np.asarray(snr_gate(), np.float64), 0.0)
            gh = self.grad.download(np.float32, self.n).astype(np.float64)
            b = 1.0 + (snr_theta * snr_theta) * V / (gh * gh + 1e-30)
            np.minimum(b, snr_cap, out=b)
            self.gate_frac = float(np.mean(b > 2.0))
            Dh = self.diag.download(np.float32, self.n).astype(np.float64)
            self.diag.upload((Dh * b).astype(np.float32))
            # diag_max was taken from the RAW diag: μ keeps its physical scale; the
            # boost enters damping, preconditioner and pred consistently via diag.
        if self.last_loss is None:
            self.last_loss = loss_fn()
        if newton_eps > 0.0:
            self._cnorm = math.sqrt(max(self._dot(self.coef, self.coef), 0.0))
        self.ctx.copy_buffer(self.coef, self.backup, nbytes)
        accepted = False; used = 0
        # batched path: fixed iterations, diagonal OR BPX preconditioner (both are
        # pure dispatches with static metas), deterministic term set (a
        # MinibatchedTerm redraws its minibatch set per step → unbatchable)
        # A term may declare `stochastic`: a BatchedRowSystem does, and says False
        # at K=1, where the drawn set is the full row set every step and the
        # captured sequence stays valid. Without that, wrapping the full solve
        # would silently cost it the capture.
        def _redraws(t):
            st = getattr(t, "stochastic", None)
            return hasattr(t, "batcher") if st is None else bool(st)
        batchable = (noise == 0.0 and newton_eps == 0.0
                     and snr_gate is None and not any(_redraws(t) for t in terms))
        for _ in range(max_tries):
            mu_abs = self.mu_rel * self.diag_max
            if batchable:
                used = self._pcg_batched(terms, mu_abs, cg_iters)
            else:
                used = self._pcg(terms, mu_abs, cg_iters, cg_tol, cg_noise=noise,
                                 newton_eps=newton_eps)
            if line_search:
                a_best, L1 = self._quartic_ls(loss_fn, self.last_loss)
                self.last_alpha = a_best
                if a_best > 0.0 and L1 < self.last_loss:
                    self.last_loss = L1
                    # μ adapts from the accepted step length: full steps ⇒ the damped
                    # model is trustworthy (go bolder); tiny steps ⇒ damp harder.
                    if a_best >= 0.8:
                        self.mu_rel = max(self.mu_rel / 3.0, mu_rel_min)
                    elif a_best <= 0.25:
                        self.mu_rel = min(self.mu_rel * 3.0, mu_rel_max)
                    self.nu = 2.0
                    accepted = True
                    break
                self.ctx.copy_buffer(self.backup, self.coef, nbytes)   # no descent on the ray
                self.mu_rel = min(self.mu_rel * self.nu, mu_rel_max); self.nu *= 2.0
                continue
            # predicted reduction of the GN model:  ½·(μ·δᵀDδ − gᵀδ)
            gd = self._dot(self.grad, self.delta)
            self.z.zero(); self._madd_diag(self.delta, self.diag, 1.0, self.z)  # z = D∘δ
            dDd = self._dot(self.delta, self.z)
            pred = 0.5 * (mu_abs * dDd - gd)
            self._axpby(self.delta, self.backup, 1.0, 1.0, self.coef)   # coef = backup + δ
            L1 = loss_fn()
            actual = self.last_loss - L1
            if np.isfinite(L1) and actual > 0.0 and pred > 0.0:
                rho = actual / pred
                self.mu_rel = max(self.mu_rel * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3),
                                  mu_rel_min)
                self.nu = 2.0
                self.last_loss = L1
                self.last_pred = float(pred); self.last_actual = float(actual)
                accepted = True
                break
            self.mu_rel = min(self.mu_rel * self.nu, mu_rel_max); self.nu *= 2.0
            self.ctx.copy_buffer(self.backup, self.coef, nbytes)        # restore, re-solve
        return self.last_loss, accepted, used

