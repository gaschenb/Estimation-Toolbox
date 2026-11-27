"""MCMC samplers on a log density, vectorized across chains.

All samplers take `logp_batch(X) -> (C,)` evaluating C points (C, d) at once (a batched
forward model makes every chain step a single call). HMC also needs
`grad_batch(X) -> (logp (C,), grad (C, d))`.

  rwm        random-walk Metropolis, fixed proposal (2.38²/d)·Σ
  am         adaptive Metropolis (Haario, Saksman & Tamminen 2001) with Robbins–Monro
             scale adaptation to acceptance 0.234; adaptation is frozen after burn-in so
             the retained chain is a time-homogeneous Markov chain
  ensemble   affine-invariant stretch move (Goodman & Weare 2010), two half-ensembles
  hmc        Hamiltonian Monte Carlo, Euclidean metric M⁻¹ = Σ (e.g. Laplace covariance),
             dual-averaging step size (Hoffman & Gelman 2014), jittered path length
  pmmh       (sysid.pmmh) pseudo-marginal MH on particle-filter likelihoods
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .diagnostics import summarize
from .linalg import chol


@dataclass
class MCMCResult:
    method: str
    chains: np.ndarray          # (C, N, d) post burn-in
    logp: np.ndarray            # (C, N)
    accept_rate: float
    runtime: float
    n_evals: int
    extras: dict = field(default_factory=dict)

    @property
    def samples(self):
        return self.chains.reshape(-1, self.chains.shape[-1])

    def summary(self):
        s = summarize(self.chains)
        s["ess_per_s"] = s["ess_bulk"] / max(self.runtime, 1e-12)
        return s


def _init_chains(x0, cov, C, rng, logp_batch, spread=1.0):
    x0 = np.asarray(x0, float)
    if x0.ndim == 2:
        X = x0.copy()
    else:
        L = chol(cov)
        X = x0 + spread * rng.standard_normal((C, x0.size)) @ L.T
        X[0] = x0
    lp = logp_batch(X)
    for _ in range(50):          # re-draw any start outside the support
        bad = ~np.isfinite(lp)
        if not bad.any():
            break
        X[bad] = x0 + 0.1 * spread * rng.standard_normal((bad.sum(), x0.size)) @ chol(cov).T
        lp[bad] = logp_batch(X[bad])
    return X, lp


def rwm(logp_batch, x0, cov, n_iter=5000, burn=1000, n_chains=4, rng=None, scale=None):
    rng = np.random.default_rng(rng)
    d = np.size(x0) if np.ndim(x0) == 1 else np.shape(x0)[1]
    scale = 2.38 / np.sqrt(d) if scale is None else scale
    L = scale * chol(cov)
    t0 = time.perf_counter()
    X, lp = _init_chains(x0, cov, n_chains, rng, logp_batch)
    chains = np.empty((n_chains, n_iter - burn, d)); lps = np.empty((n_chains, n_iter - burn))
    acc = 0
    for i in range(n_iter):
        Y = X + rng.standard_normal(X.shape) @ L.T
        lq = logp_batch(Y)
        a = np.log(rng.random(n_chains)) < lq - lp
        X[a], lp[a] = Y[a], lq[a]
        if i >= burn:
            chains[:, i - burn], lps[:, i - burn] = X, lp
            acc += a.sum()
    return MCMCResult("rwm", chains, lps, acc / (n_chains * (n_iter - burn)),
                      time.perf_counter() - t0, n_chains * (n_iter + 1))


def am(logp_batch, x0, cov, n_iter=5000, burn=1500, n_chains=4, rng=None, target=0.234,
       name="am"):
    rng = np.random.default_rng(rng)
    d = np.size(x0) if np.ndim(x0) == 1 else np.shape(x0)[1]
    t0 = time.perf_counter()
    X, lp = _init_chains(x0, cov, n_chains, rng, logp_batch)
    mu = X.mean(0).copy()
    Sig = np.asarray(cov, float).copy()
    logs = np.log(2.38**2 / d)
    eps = 1e-10 * np.mean(np.diag(Sig))
    chains = np.empty((n_chains, n_iter - burn, d)); lps = np.empty((n_chains, n_iter - burn))
    acc, n_seen = 0, 1
    L = np.exp(0.5 * logs) * chol(Sig)
    for i in range(n_iter):
        if i < burn and i % 10 == 0:
            L = np.exp(0.5 * logs) * chol(Sig + eps * np.eye(d))
        Y = X + rng.standard_normal(X.shape) @ L.T
        lq = logp_batch(Y)
        la = np.minimum(0.0, lq - lp)
        la[~np.isfinite(la)] = -np.inf
        a = np.log(rng.random(n_chains)) < la
        X[a], lp[a] = Y[a], lq[a]
        if i < burn:     # adaptation (pooled over chains); frozen afterwards
            g = 1.0 / (i + 1) ** 0.6
            logs += g * (np.mean(np.exp(la)) - target)
            for x in X:
                n_seen += 1
                w = 1.0 / n_seen
                dx = x - mu
                mu = mu + w * dx
                Sig = Sig + w * (np.outer(dx, x - mu) - Sig)
        else:
            chains[:, i - burn], lps[:, i - burn] = X, lp
            acc += a.sum()
    return MCMCResult(name, chains, lps, acc / (n_chains * (n_iter - burn)),
                      time.perf_counter() - t0, n_chains * (n_iter + 1),
                      extras={"scale": float(np.exp(logs)), "proposal_cov": Sig})


def ensemble(logp_batch, x0, cov, n_iter=3000, burn=1000, n_walkers=None, rng=None, a=2.0):
    rng = np.random.default_rng(rng)
    d = np.size(x0) if np.ndim(x0) == 1 else np.shape(x0)[1]
    W = n_walkers or max(4 * d, 16)
    W += W % 2
    t0 = time.perf_counter()
    X, lp = _init_chains(x0, cov, W, rng, logp_batch, spread=0.5)
    chains = np.empty((W, n_iter - burn, d)); lps = np.empty((W, n_iter - burn))
    half = W // 2
    sets = (np.arange(half), np.arange(half, W))
    acc, nev = 0, W
    for i in range(n_iter):
        for s in (0, 1):
            act, oth = sets[s], sets[1 - s]
            z = ((a - 1.0) * rng.random(half) + 1.0) ** 2 / a       # g(z) ∝ 1/√z on [1/a, a]
            Xj = X[oth[rng.integers(0, half, half)]]
            Y = Xj + z[:, None] * (X[act] - Xj)
            lq = logp_batch(Y)
            nev += half
            la = (d - 1) * np.log(z) + lq - lp[act]
            ok = np.log(rng.random(half)) < la
            idx = act[ok]
            X[idx], lp[idx] = Y[ok], lq[ok]
            if i >= burn:
                acc += ok.sum()
        if i >= burn:
            chains[:, i - burn], lps[:, i - burn] = X, lp
    return MCMCResult("ensemble", chains, lps, acc / (W * (n_iter - burn)),
                      time.perf_counter() - t0, nev, extras={"n_walkers": W})


def hmc(grad_batch, x0, cov, n_iter=2000, burn=500, n_chains=4, rng=None, path_len=None,
        target=0.8, max_leapfrog=100):
    """Euclidean HMC with mass matrix M = cov⁻¹ (so momentum p ~ N(0, cov⁻¹))."""
    rng = np.random.default_rng(rng)
    d = np.size(x0) if np.ndim(x0) == 1 else np.shape(x0)[1]
    cov = np.asarray(cov, float)
    Lc = chol(cov)                          # x-space metric: velocity = cov · p
    Linv = np.linalg.inv(Lc)                # p = Linv^T ξ  ⇒ p ~ N(0, cov⁻¹)
    path_len = 1.5 if path_len is None else path_len     # in whitened units
    t0 = time.perf_counter()
    X, _ = _init_chains(x0, cov, n_chains, rng, lambda Z: grad_batch(Z)[0])
    lp, g = grad_batch(X)
    nev = n_chains
    # dual averaging state
    eps = 0.5
    mu, Hbar, logeb, t0da, gam, kap = np.log(10 * eps), 0.0, 0.0, 10.0, 0.05, 0.75
    chains = np.empty((n_chains, n_iter - burn, d)); lps = np.empty((n_chains, n_iter - burn))
    acc_sum, n_div = 0.0, 0
    for i in range(n_iter):
        e = eps if i < burn else np.exp(logeb)
        e_i = e * rng.uniform(0.8, 1.2)
        nL = int(np.clip(np.ceil(path_len / e_i), 1, max_leapfrog))
        P = rng.standard_normal((n_chains, d)) @ Linv          # p = ξ L^{-1} (row form)
        K0 = 0.5 * np.sum((P @ Lc) ** 2, axis=1)              # ½ pᵀ cov p
        Xn, Pn, gn = X.copy(), P + 0.5 * e_i * g, g
        for j in range(nL):
            Xn = Xn + e_i * (Pn @ cov)
            lpn, gn = grad_batch(Xn)
            nev += n_chains
            if j < nL - 1:
                Pn = Pn + e_i * gn
        Pn = Pn + 0.5 * e_i * gn
        K1 = 0.5 * np.sum((Pn @ Lc) ** 2, axis=1)
        logr = (lpn - K1) - (lp - K0)
        logr[~np.isfinite(logr)] = -np.inf
        n_div += int(np.sum(logr < -1000))
        acc = np.exp(np.minimum(0.0, logr))
        ok = np.log(rng.random(n_chains)) < logr
        X[ok], lp[ok], g[ok] = Xn[ok], lpn[ok], gn[ok]
        if i < burn:
            m = i + 1
            Hbar = (1 - 1 / (m + t0da)) * Hbar + (target - acc.mean()) / (m + t0da)
            loge = mu - np.sqrt(m) / gam * Hbar
            eps = np.exp(loge)
            w = m ** (-kap)
            logeb = w * loge + (1 - w) * logeb
        else:
            chains[:, i - burn], lps[:, i - burn] = X, lp
            acc_sum += acc.sum()
    return MCMCResult("hmc", chains, lps, acc_sum / (n_chains * (n_iter - burn)),
                      time.perf_counter() - t0, nev,
                      extras={"step_size": float(np.exp(logeb)), "divergences": n_div})
