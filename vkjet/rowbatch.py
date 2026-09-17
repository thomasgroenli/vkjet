"""Minibatching the row system: the full solve as the K=1 special case.

The solver's contract is least-squares on rows. Minibatching is therefore not a
data-side optimisation bolted onto a physics-side deterministic core — it is the
general form of the same contract, and the full solve is K=1. That distinction
is load-bearing: the older `optim.MinibatchedTerm` could only work by being
ordered LAST among the terms, with wall+PDE already summed into `grad_buf` as a
deterministic baseline. That baseline is exactly the privileged-row split the
architecture rejects, so here there is none: every term is bucketed, and the
norm test measures the variance of the whole system's gradient estimate.

WHAT A BUCKET IS. One random partition of the row set into K parts, drawn once
per outer step (`plan`), each part a persistent device sub-batch on every term.
An outer step draws buckets in random order until the norm test passes, so the
effective batch self-calibrates: oversampled rows need a few buckets, a
near-optimal iterate needs most of them.

  * The estimator is EXACTLY unbiased for any bucket sizes. Each bucket has
    probability K_used/K of being drawn, so E[Σ_{k in used} g_k] = (K_used/K)·g,
    and `scale = K/K_used` inverts it. Bucket sizes differ by at most one row;
    the only thing that costs is the sample-VARIANCE estimate feeding the norm
    test, which is a tolerance, not the objective.
  * The norm test carries the FINITE-POPULATION correction (1 - K_used/K), and
    that is a consequence of the contract, not a refinement of the statistics.
    The classical Byrd-Bollapragada test omits it because it reads the rows as a
    sample from a wider distribution, so even the full batch retains sampling
    error. Here the objective IS the sum over these rows: drawing all K buckets
    reproduces it exactly, so the variance that remains must be zero. Measured
    on the production operator set the difference is not cosmetic — the
    uncorrected test reports sigma/|g| ~ 0.39 at FULL batch and therefore never
    passes at tol=0.15, which reads as "your data is noise" when what actually
    happened is that the estimator was compared against a population that does
    not exist.

    This is also why there is no `snr_fail` here, unlike optim.MinibatchedTerm:
    under the contract the full-batch gradient cannot fail its own SNR test. It
    is not estimating anything. A stopping rule has to come from the objective's
    own stabilisation (rowfit's stop_rel), and the floor it sits above is the
    arithmetic one at K=1 and the statistical one for a partial draw — one
    criterion, one floor, two regimes.
  * CG replays the SAME bucket set for every iteration, so the Krylov space is
    built for one consistent operator. The Newton finite-difference taps replay
    it too (`accumulate_replay`).
  * diag and loss always use the FULL row set. diag is the preconditioner (a
    stochastic M would make CG non-stationary for no benefit), and loss is the
    real objective — LM accept/reject and the `stop_rel` convergence test both
    read it, and both would be measuring the wrong thing on a subsample.

K=1 IS AN ALIAS, NOT A CODE PATH. With one bucket the sub-batch IS the full
batch object (no copy, no extra memory), `scale` is exactly 1.0, and
`accumulate`/`hvp` issue exactly the dispatches the unwrapped term list would.
That is what makes "the full solve is a special case" a fact about the code
rather than a claim about the design. tests/test_minibatch_rows.py checks it.
"""
from __future__ import annotations

import math

import numpy as np

from .context import Context
from .optim import NormTestBatcher


def row_buckets(n_rows, k, seed=0):
    """Partition range(n_rows) into `k` buckets of near-equal size.

    Strided slices of one permutation, each SORTED back into the caller's order:
    the assignment is random (what the estimator needs) while each bucket stays
    in the caller's spatially coherent record order (what the gather/scatter
    kernels need — a shuffled layout costs ~3x on the atomics). k=1 returns
    exactly [arange(n_rows)], the identity.
    """
    n = int(n_rows); k = max(1, int(k))
    if k == 1:
        return [np.arange(n)]
    perm = np.random.default_rng(seed).permutation(n)
    return [np.sort(perm[i::k]) for i in range(k)]


def local_buckets(of_row, sel, k):
    """Bucket membership expressed in a term's OWN row order.

    `of_row` is the global per-row bucket id; `sel` the global row indices the
    term was bound with, in binding order. Returns k index arrays into the
    term's local rows.
    """
    b = np.asarray(of_row)[np.asarray(sel)]
    return [np.flatnonzero(b == i) for i in range(k)]


class BatchedRowSystem:
    """The whole row system as ONE term, with the minibatch hidden inside.

    Presents the standard 4-method term protocol to GaussNewtonCG
    (accumulate / hvp / accumulate_diag / loss), so the optimizer stays blind to
    the fact that it is being fed an estimate — and blind, as it must be, to
    what any row MEANS. Every wrapped term must be bucketed to the same K.
    """

    def __init__(self, ctx: Context, terms, n_params, tol: float = 0.15,
                 seed: int = 0, coord_var: bool = False):
        ks = {len(getattr(t, "minibatches", ()) or ()) for t in terms}
        if len(ks) != 1 or 0 in ks:
            raise ValueError(
                "every term must be bucketed to the same K before wrapping "
                f"(got {sorted(ks)}); an unbucketed term would re-introduce the "
                "deterministic-baseline split")
        self.ctx = ctx
        self.terms = list(terms)
        self.n = int(n_params)
        self.K = ks.pop()
        self.tol = float(tol)
        self.rng = np.random.default_rng(seed)
        self.batcher = NormTestBatcher(ctx, self.n, coord_var=coord_var)
        self.scratch = ctx.buffer(self.n * 4)
        self._used = list(range(self.K))
        self._scale = 1.0
        self._token = object()          # CG capture key; see _batch
        # every bucket set partitions its term's full batch ⇒ the all-buckets
        # case is the full batch and can be issued as one dispatch per term
        self._full_equiv = all(
            sum(b["n"] for b in t.minibatches) == t._batch["n"]
            for t in self.terms)
        self.grad_noise = 0.0           # sqrt(Var(g_hat)): the CG forcing scale
        self.stochastic = self.K > 1    # K=1 draws the same (full) set forever,
                                        # so the captured CG solve stays valid
        self.newton_fd = any(getattr(t, "newton_fd", False) for t in self.terms)

    # -- the CG command-sequence capture key ------------------------------- #
    @property
    def _batch(self):
        """GaussNewtonCG keys its recorded CommandSequence on id(term._batch).
        The wrapper's dispatch chain depends on which buckets were drawn, so the
        token changes whenever the drawn set does — otherwise a stale recording
        would be replayed against a different operator. At K=1 the set never
        changes, so the capture (and its ~2x on small fits) survives."""
        return self._token

    def _all_drawn(self):
        return self._full_equiv and len(self._used) == self.K

    def set_seed(self, seed):
        for t in self.terms:
            if hasattr(t, "set_seed"):
                t.set_seed(seed)

    # -- gradient: the norm test over the WHOLE system --------------------- #
    def accumulate(self, coef_buf, grad_buf):
        if self.K == 1:
            for t in self.terms:
                t.accumulate(coef_buf, grad_buf)
            self._used, self._scale = [0], 1.0
            self.grad_noise = 0.0
            return
        b = self.batcher
        order = self.rng.permutation(self.K)
        b.M.zero()
        if b.coordv:
            b.Q.zero()
        b.last_var = None; b.last_norm2 = None; b.passed = False
        Q = 0.0; used = []
        for k in order:
            b.gk.zero()
            for t in self.terms:
                t.accumulate(coef_buf, b.gk, batch=t.minibatches[k])
            Q += b._dot(b.gk, b.gk)
            if b.coordv:
                b.ctx.run(b.madd_p, [b.gk, b.gk, b.Q, b._vm(1.0, 0.0)],
                          groups=b._g())
            b._axpby(b.gk, b.M, 1.0, 1.0, b.M)
            used.append(int(k)); K = len(used)
            if K >= 2 and (self.tol > 0.0 or K == self.K):
                mM = b._dot(b.M, b.M)
                # Var(g_hat) for sampling K of self.K buckets WITHOUT
                # replacement: (K_tot^2/K)*S^2*(1 - K/K_tot). The last factor is
                # the finite-population correction; it is what makes the test
                # pass exactly when the whole row set has been drawn.
                s2 = (Q - mM / K) / (K - 1)
                var = (float(self.K * self.K) / K) * s2 * (1.0 - K / self.K)
                b.last_var = var
                b.last_norm2 = float(self.K * self.K) / (K * K) * mM
                if self.tol > 0.0 and var <= self.tol * self.tol * b.last_norm2:
                    b.passed = True
                    break
        b.exhausted = len(used) == self.K
        b._K_used, b._nmb = len(used), self.K
        prev, self._used = tuple(self._used), used
        self._scale = float(self.K) / len(used)
        if tuple(used) != prev:
            self._token = object()          # invalidate the captured CG solve
        b._axpby(b.M, grad_buf, self._scale, 1.0, grad_buf)
        self.grad_noise = (math.sqrt(max(b.last_var, 0.0))
                           if b.last_var is not None else 0.0)

    # -- curvature action: the SAME buckets CG's gradient came from -------- #
    def hvp(self, coef_buf, v_buf, out_buf):
        if self._all_drawn():
            for t in self.terms:
                t.hvp(coef_buf, v_buf, out_buf)
            return
        self.scratch.zero()
        for k in self._used:
            for t in self.terms:
                t.hvp(coef_buf, v_buf, self.scratch, batch=t.minibatches[k])
        self.batcher._axpby(self.scratch, out_buf, self._scale, 1.0, out_buf)

    def accumulate_replay(self, coef_buf, grad_buf):
        """Gradient at an ARBITRARY coef over the last drawn bucket set — the
        Newton finite-difference taps must see the same stochastic operator CG
        is solving. Does not re-run the norm test."""
        if self._all_drawn():
            for t in self.terms:
                t.accumulate(coef_buf, grad_buf)
            return
        self.scratch.zero()
        for k in self._used:
            for t in self.terms:
                t.accumulate(coef_buf, self.scratch, batch=t.minibatches[k])
        self.batcher._axpby(self.scratch, grad_buf, self._scale, 1.0, grad_buf)

    # -- full row set: preconditioner and the true objective --------------- #
    def accumulate_diag(self, coef_buf, diag_buf):
        for t in self.terms:
            t.accumulate_diag(coef_buf, diag_buf)

    def loss(self, coef_buf, loss_buf):
        for t in self.terms:
            t.loss(coef_buf, loss_buf)

    @property
    def eff_batch(self):
        return (len(self._used), self.K)

    def coord_var_host(self):
        return self.batcher.coord_var_host()
