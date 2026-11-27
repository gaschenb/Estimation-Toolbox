"""Sequential Monte Carlo.

  pf       bootstrap (SIR) particle filter
  rpf      regularized PF: after resampling, particles are jittered with a Gaussian kernel
           of covariance h²·Ĉov, h = (4/(n+2))^{1/(n+4)} N^{−1/(n+4)} (Musso, Oudjane &
           Le Gland 2001); counters sample impoverishment when Q is small
  pf-ekf   particle filter with the locally linearized optimal proposal
           q(x_k | x_{k-1}, y_k) = N(μ_i, Σ_i), obtained by linearizing h at f(x_{k-1}^i)
           (Doucet, Godsill & Andrieu 2000). Requires an additive Gaussian measurement.

Resampling (multinomial / stratified / systematic / residual) is triggered when
ESS < threshold·N. The likelihood estimate Σ log(Σ_i w̃_{k-1}^i α_k^i) is the
standard unbiased (in p, not log p) SMC estimator used by PMMH.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from .linalg import chol, sym


def logsumexp(a):
    c = np.max(a)
    if not np.isfinite(c):
        return c
    return c + np.log(np.sum(np.exp(a - c)))


def resample(logw_or_w, rng, scheme="systematic"):
    w = np.asarray(logw_or_w, float)
    N = w.size
    w = w / w.sum()
    if scheme == "multinomial":
        return rng.choice(N, size=N, p=w)
    if scheme == "residual":
        nk = np.floor(N * w).astype(int)
        idx = np.repeat(np.arange(N), nk)
        R = N - nk.sum()
        if R > 0:
            wr = N * w - nk
            wr /= wr.sum()
            idx = np.concatenate([idx, rng.choice(N, size=R, p=wr)])
        return idx
    if scheme == "stratified":
        u = (np.arange(N) + rng.random(N)) / N
    elif scheme == "systematic":
        u = (np.arange(N) + rng.random()) / N
    else:
        raise ValueError(scheme)
    cs = np.cumsum(w)
    cs[-1] = 1.0
    return np.searchsorted(cs, u)


@dataclass
class PFResult:
    method: str
    m: np.ndarray            # (T+1, nx) weighted means
    P: np.ndarray            # (T+1, nx, nx)
    ess: np.ndarray          # (T,) ESS before resampling
    loglik: float
    runtime: float
    particles: np.ndarray | None = None   # (T+1, N, nx) filtering particles
    logw: np.ndarray | None = None        # (T+1, N) normalized log weights
    n_resample: int = 0
    v: np.ndarray | None = None           # predictive innovations (Gaussian obs only)
    S: np.ndarray | None = None
    diverged: bool = False
    extras: dict = field(default_factory=dict)

    @property
    def std(self):
        return np.sqrt(np.maximum(np.diagonal(self.P, axis1=1, axis2=2), 0.0))


def run_pf(model, ys, n_particles=1000, proposal="bootstrap", scheme="systematic",
           ess_threshold=0.5, rng=None, store=True, regularize=False):
    rng = np.random.default_rng(rng)
    T, nx, N = len(ys), model.nx, int(n_particles)
    if proposal == "ekf" and not model.gaussian_obs:
        raise ValueError("pf-ekf needs an additive Gaussian measurement model")
    m = np.zeros((T + 1, nx)); P = np.zeros((T + 1, nx, nx)); ess = np.zeros(T)
    parts = np.zeros((T + 1, N, nx)) if store else None
    lws = np.zeros((T + 1, N)) if store else None
    gauss = model.gaussian_obs and store
    v = np.zeros((T, model.ny)) if gauss else None
    S = np.zeros((T, model.ny, model.ny)) if gauss else None

    X = model.sample_x0(rng, N)
    logw = np.full(N, -np.log(N))
    m[0], P[0] = model.m0, model.P0
    if store:
        parts[0], lws[0] = X, logw
    ll, nres = 0.0, 0
    Qinv = np.linalg.inv(model.Q)
    LQ = chol(model.Q)
    t0 = time.perf_counter()
    for k in range(1, T + 1):
        y = ys[k - 1]
        w = np.exp(logw)                 # logw is normalized
        e = 1.0 / np.sum(w * w)
        ess[k - 1] = e
        if e < ess_threshold * N:
            idx = resample(w, rng, scheme)
            if regularize:
                mu = w @ X
                d = X - mu
                Lk = chol(sym((w[:, None] * d).T @ d) + 1e-12 * np.eye(nx))
                hopt = (4.0 / (nx + 2)) ** (1.0 / (nx + 4)) * N ** (-1.0 / (nx + 4))
                X = X[idx] + hopt * rng.standard_normal((N, nx)) @ Lk.T
            else:
                X = X[idx]
            logw = np.full(N, -np.log(N))
            nres += 1
        FX = model.f(X, k)
        if gauss:   # predictive moments for NIS (before the update)
            wp = np.exp(logw)
            Z = model.h(FX + rng.standard_normal(FX.shape) @ LQ.T, k)
            zb = wp @ Z
            dZ = model.residual_y(Z, zb)
            v[k - 1] = model.residual_y(y, zb)
            S[k - 1] = sym((wp[:, None] * dZ).T @ dZ + model.R)
        if proposal == "bootstrap":
            X = FX + rng.standard_normal(FX.shape) @ LQ.T
            inc = model.obs_loglik(y, X, k)
        elif proposal == "ekf":
            H = model.H_batch(FX, k)                         # (N, ny, nx)
            Hq = H @ model.Q                                  # (N, ny, nx)
            Sg = sym(Hq @ np.swapaxes(H, 1, 2) + model.R)     # (N, ny, ny)
            Kg = np.swapaxes(np.linalg.solve(Sg, Hq), 1, 2)   # (N, nx, ny)
            hF = model.h(FX, k)
            d = model.residual_y(y, hF)                       # (N, ny)
            mu = FX + np.einsum("nij,nj->ni", Kg, d)
            Sig = sym(model.Q - Kg @ Hq)
            L = chol(Sig)
            Eps = rng.standard_normal((N, nx))
            X = mu + np.einsum("nij,nj->ni", L, Eps)
            # log weights: p(y|x) p(x|x_prev) / q(x)
            dx = X - FX
            lp_trans = -0.5 * np.einsum("ni,ij,nj->n", dx, Qinv, dx)
            lq = -0.5 * np.sum(Eps * Eps, 1) - np.sum(np.log(np.diagonal(L, axis1=1, axis2=2)), 1)
            lp_trans -= np.sum(np.log(np.diag(LQ)))
            inc = model.obs_loglik(y, X, k) + lp_trans - lq
        else:
            raise ValueError(proposal)
        logw = logw + inc
        c = logsumexp(logw)
        if not np.isfinite(c):
            m[k:] = np.nan; P[k:] = np.nan
            return PFResult(("rpf" if regularize else "pf") if proposal == "bootstrap" else "pf-ekf", m, P, ess, -np.inf,
                            time.perf_counter() - t0, parts, lws, nres, v, S, diverged=True)
        ll += c          # logw was normalized before the increment
        logw = logw - c
        if store:
            w = np.exp(logw)
            m[k] = w @ X
            d = X - m[k]
            P[k] = sym((w[:, None] * d).T @ d)
            parts[k], lws[k] = X, logw
    name = ("rpf" if regularize else "pf") if proposal == "bootstrap" else "pf-ekf"
    return PFResult(name, m, P, ess, ll, time.perf_counter() - t0, parts, lws, nres, v, S,
                    extras={"n_particles": N, "scheme": scheme})


def pf_loglik(model, ys, n_particles=500, rng=None, proposal="bootstrap"):
    """Likelihood estimate only (no storage), for PMMH."""
    return run_pf(model, ys, n_particles, proposal=proposal, rng=rng, store=False,
                  ess_threshold=1.0 if proposal == "bootstrap" else 0.5).loglik
