"""Smoothers.

  rts     Gaussian Rauch–Tung–Striebel smoother for any FilterResult. With the
          stored cross-covariance C_k = Cov(x_{k-1}, x_k | y_{1:k-1}),
              G_k = C_{k+1} (P⁻_{k+1})^{-1}
              m^s_k = m_k + G_k (m^s_{k+1} − m⁻_{k+1})
              P^s_k = P_k + G_k (P^s_{k+1} − P⁻_{k+1}) G_kᵀ
          this is the RTS / ERTS / URTS / cubature / ensemble RTS smoother
          depending on which filter produced C. Also returns the lag-one
          smoothed cross-covariance P^s_{k+1,k} = P^s_{k+1} G_kᵀ (used by EM).
  ffbsi   forward-filtering backward-simulation particle smoother
          (Godsill, Doucet & West 2004), vectorized O(T·M·N).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from .linalg import solve_psd, sym


@dataclass
class SmootherResult:
    method: str
    m: np.ndarray           # (T+1, nx)
    P: np.ndarray           # (T+1, nx, nx)
    runtime: float
    G: np.ndarray | None = None
    Pcross: np.ndarray | None = None   # Pcross[k] = Cov(x_{k+1}, x_k | y_{1:T}), k = 0..T-1
    samples: np.ndarray | None = None  # (M, T+1, nx) for particle smoothers
    extras: dict = field(default_factory=dict)

    @property
    def std(self):
        return np.sqrt(np.maximum(np.diagonal(self.P, axis1=1, axis2=2), 0.0))


def rts(fr):
    t0 = time.perf_counter()
    T1, nx = fr.m.shape
    ms = fr.m.copy(); Ps = fr.P.copy()
    if fr.diverged or not np.all(np.isfinite(fr.Pp[1:])):
        nan = np.full_like(ms, np.nan)
        return SmootherResult(fr.method + "+rts", nan, np.full_like(Ps, np.nan), 0.0)
    G = np.zeros((T1 - 1, nx, nx))
    Pc = np.zeros((T1 - 1, nx, nx))
    for k in range(T1 - 2, -1, -1):
        Gk = solve_psd(fr.Pp[k + 1], fr.C[k + 1].T).T
        ms[k] = fr.m[k] + Gk @ (ms[k + 1] - fr.mp[k + 1])
        Ps[k] = sym(fr.P[k] + Gk @ (Ps[k + 1] - fr.Pp[k + 1]) @ Gk.T)
        G[k] = Gk
        Pc[k] = Ps[k + 1] @ Gk.T
    return SmootherResult(fr.method + "+rts", ms, Ps, time.perf_counter() - t0, G, Pc)


def ffbsi(pf, model, n_traj=100, rng=None):
    if pf.particles is None:
        raise ValueError("ffbsi needs a particle filter run with store=True")
    rng = np.random.default_rng(rng)
    t0 = time.perf_counter()
    X, LW = pf.particles, pf.logw
    T1, N, nx = X.shape
    M = int(n_traj)
    LQinv = np.linalg.inv(np.linalg.cholesky(model.Q))
    traj = np.zeros((M, T1, nx))
    w = np.exp(LW[-1] - logsumexp(LW[-1]))
    idx = rng.choice(N, size=M, p=w)
    traj[:, -1] = X[-1, idx]
    for k in range(T1 - 2, -1, -1):
        FX = model.f(X[k], k + 1)                         # (N, nx)
        d = traj[:, k + 1][:, None, :] - FX[None]          # (M, N, nx)
        z = d @ LQinv.T
        lw = LW[k][None, :] - 0.5 * np.sum(z * z, axis=2)  # (M, N)
        lw -= logsumexp(lw, axis=1, keepdims=True)
        u = rng.random((M, 1))
        j = np.minimum((np.cumsum(np.exp(lw), axis=1) < u).sum(1), N - 1)
        traj[:, k] = X[k, j]
    ms = traj.mean(0)
    d = traj - ms
    Ps = np.einsum("mti,mtj->tij", d, d) / max(M - 1, 1)
    return SmootherResult(pf.method + "+ffbsi", ms, Ps, time.perf_counter() - t0, samples=traj)
