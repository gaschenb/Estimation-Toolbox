"""Ensemble Kalman filters.

  enkf   stochastic EnKF with perturbed observations (Burgers et al. 1998)
  etkf   ensemble transform Kalman filter, symmetric square root (Hunt et al. 2007)

Both support multiplicative inflation of forecast anomalies. The sample
cross-covariance between the previous analysis and the forecast ensemble is
stored as C, so smoothers.rts gives an ensemble RTS smoother.
"""
from __future__ import annotations

import time

import numpy as np

from .gaussian import FilterResult
from .linalg import chol, gauss_logpdf, solve_psd, sym


def run_enkf(model, ys, n_ens=100, kind="enkf", inflation=1.0, rng=None):
    if not model.gaussian_obs:
        raise ValueError("EnKF needs an additive Gaussian measurement model")
    rng = np.random.default_rng(rng)
    T, nx, ny = len(ys), model.nx, model.ny
    N = int(n_ens)
    m = np.zeros((T + 1, nx)); P = np.zeros((T + 1, nx, nx))
    mp = np.zeros_like(m); Pp = np.zeros_like(P); C = np.zeros_like(P)
    v = np.zeros((T, ny)); S = np.zeros((T, ny, ny))
    m[0] = mp[0] = model.m0; P[0] = Pp[0] = model.P0
    X = model.sample_x0(rng, N)
    LR = chol(model.R)
    Rinv_half = np.linalg.inv(LR)
    ll, diverged = 0.0, False
    err = np.seterr(all="ignore")
    t0 = time.perf_counter()
    for k in range(1, T + 1):
        Xa = X
        X = model.sample_trans(X, k, rng)
        xb = X.mean(0)
        A = X - xb
        if inflation != 1.0:
            A = A * inflation
            X = xb + A
        mp[k], Pp[k] = xb, sym(A.T @ A / (N - 1))
        C[k] = (Xa - Xa.mean(0)).T @ A / (N - 1)
        Z = model.h(X, k)
        zb = Z.mean(0)
        dZ = model.residual_y(Z, zb)
        y = ys[k - 1]
        Szz = sym(dZ.T @ dZ / (N - 1) + model.R)
        vk = model.residual_y(y, zb)
        v[k - 1], S[k - 1] = vk, Szz
        if kind == "enkf":
            Pxz = A.T @ dZ / (N - 1)
            K = solve_psd(Szz, Pxz.T).T
            E = rng.standard_normal((N, ny)) @ LR.T
            E -= E.mean(0)                       # centred perturbations
            X = X + model.residual_y(y + E, Z) @ K.T
        elif kind == "etkf":
            Ys = dZ @ Rinv_half.T / np.sqrt(N - 1)      # (N, ny) whitened anomalies
            # thin SVD Ys = U Σ Wᵀ:  (I + Ys Ysᵀ)^{-p} = I + U((1 + Σ²)^{-p} − 1)Uᵀ, O(N·ny²)
            U, sv, _ = np.linalg.svd(Ys, full_matrices=False)
            lam = 1.0 + sv**2
            g = Ys @ (Rinv_half @ vk) / np.sqrt(N - 1)
            wbar = g + U @ ((1.0 / lam - 1.0) * (U.T @ g))
            TA = A + U @ ((1.0 / np.sqrt(lam) - 1.0)[:, None] * (U.T @ A))   # T·A, T symmetric
            X = xb + wbar @ A + TA
        else:
            raise ValueError(kind)
        m[k] = X.mean(0)
        dA = X - m[k]
        P[k] = sym(dA.T @ dA / (N - 1))
        try:
            ll += gauss_logpdf(vk, Szz)[0]
        except np.linalg.LinAlgError:
            pass
        if not np.all(np.isfinite(X)):
            diverged = True
            m[k:] = np.nan; P[k:] = np.nan; mp[k:] = np.nan; Pp[k:] = np.nan
            break
    np.seterr(**err)
    return FilterResult(kind, m, P, mp, Pp, C, v, S, ll, time.perf_counter() - t0, diverged,
                        extras={"n_ens": N, "inflation": inflation})
