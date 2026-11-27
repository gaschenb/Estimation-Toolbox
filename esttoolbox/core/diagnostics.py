"""MCMC convergence diagnostics (Vehtari, Gelman, Simpson, Carpenter & Bürkner 2021).

rhat      rank-normalized split-R̂ (max of bulk and folded versions)
ess_bulk  multi-chain ESS of rank-normalized draws, Geyer initial monotone sequence
ess_tail  min ESS of the 5 % / 95 % quantile indicators
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata


def _split(x):
    """(M, N) -> (2M, N//2)."""
    M, N = x.shape
    h = N // 2
    return np.vstack([x[:, :h], x[:, N - h:]])


def _rank_normalize(x):
    S = x.size
    r = rankdata(x, method="average").reshape(x.shape)
    return norm.ppf((r - 3 / 8) / (S + 1 / 4))


def _rhat_basic(x):
    M, N = x.shape
    if N < 2:
        return np.nan
    W = x.var(axis=1, ddof=1).mean()
    B = N * x.mean(axis=1).var(ddof=1) if M > 1 else 0.0
    var_plus = (N - 1) / N * W + B / N
    return float(np.sqrt(var_plus / W)) if W > 0 else np.nan


def rhat(x):
    """x: (M chains, N draws) for one scalar quantity."""
    x = _split(np.asarray(x, float))
    bulk = _rhat_basic(_rank_normalize(x))
    folded = _rhat_basic(_rank_normalize(np.abs(x - np.median(x))))
    return max(bulk, folded)


def _autocov(x):
    """Per-chain autocovariance via FFT, (M, N) -> (M, N)."""
    M, N = x.shape
    x = x - x.mean(axis=1, keepdims=True)
    n = 1 << (2 * N - 1).bit_length()
    f = np.fft.rfft(x, n=n, axis=1)
    ac = np.fft.irfft(f * np.conj(f), n=n, axis=1)[:, :N] / N
    return ac


def _ess_basic(x):
    M, N = x.shape
    if N < 4:
        return np.nan
    ac = _autocov(x)
    W = (ac[:, 0] * N / (N - 1)).mean()
    var_plus = W * (N - 1) / N
    if M > 1:
        var_plus += x.mean(axis=1).var(ddof=1)
    if var_plus <= 0:
        return np.nan
    rho = 1.0 - (W - ac.mean(axis=0)) / var_plus
    rho[0] = 1.0
    # Geyer: sum of adjacent pairs while positive, enforced monotone
    P = rho[:-1:2] + rho[1::2]
    tau_sum, prev = 0.0, np.inf
    for p in P:
        if p <= 0:
            break
        p = min(p, prev)
        tau_sum += p
        prev = p
    tau = -1.0 + 2.0 * tau_sum
    tau = max(tau, 1.0 / np.log10(M * N))
    return float(M * N / tau)


def ess_bulk(x):
    return _ess_basic(_rank_normalize(_split(np.asarray(x, float))))


def ess_tail(x):
    x = _split(np.asarray(x, float))
    q05, q95 = np.quantile(x, [0.05, 0.95])
    return min(_ess_basic((x <= q05).astype(float)), _ess_basic((x <= q95).astype(float)))


def summarize(chains):
    """chains (M, N, d) -> dict of per-parameter arrays."""
    chains = np.asarray(chains, float)
    M, N, d = chains.shape
    flat = chains.reshape(-1, d)
    return {
        "mean": flat.mean(0), "std": flat.std(0, ddof=1),
        "q05": np.quantile(flat, 0.05, axis=0), "q50": np.median(flat, axis=0),
        "q95": np.quantile(flat, 0.95, axis=0),
        "rhat": np.array([rhat(chains[:, :, j]) for j in range(d)]),
        "ess_bulk": np.array([ess_bulk(chains[:, :, j]) for j in range(d)]),
        "ess_tail": np.array([ess_tail(chains[:, :, j]) for j in range(d)]),
    }
