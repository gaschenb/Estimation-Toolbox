"""Filter consistency and accuracy diagnostics (Bar-Shalom, Li & Kirubarajan 2001, §5.4).

For a consistent filter, e_k = x_k − m_k and v_k satisfy
    NEES_k = e_kᵀ P_k^{-1} e_k ~ χ²(n_x),    NIS_k = v_kᵀ S_k^{-1} v_k ~ χ²(n_y).
Single run: the time average over T steps lies in χ²_{T·n}(α/2, 1−α/2)/T.
M Monte Carlo runs: the run average at each step lies in χ²_{M·n}(·)/M.
Innovation whiteness: whitened innovations S^{-1/2} v are i.i.d. N(0, I); we report the
normalized autocorrelation (≈ N(0, 1/T)) and Ljung–Box Q statistics.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from .linalg import batch_mahalanobis, chol


def nees(xs, m, P):
    """xs, m (T+1, nx), P (T+1, nx, nx) -> (T,) NEES for k = 1..T."""
    e = xs[1:] - m[1:]
    return batch_mahalanobis(e, P[1:])


def nis(v, S):
    return batch_mahalanobis(v, S)


def chi2_band(dof, n_avg=1, alpha=0.05):
    """Two-sided (1−α) acceptance band for the average of n_avg χ²(dof) variables."""
    lo, hi = chi2.ppf([alpha / 2, 1 - alpha / 2], dof * n_avg)
    return lo / n_avg, hi / n_avg


def rmse(xs, m, axis=None):
    e = xs[1:] - m[1:]
    if axis is None:
        return float(np.sqrt(np.nanmean(np.sum(e * e, axis=1))))
    return np.sqrt(np.nanmean(e * e, axis=axis))


def whiten(v, S):
    L = chol(S)
    return np.linalg.solve(L, v[..., None])[..., 0]


def autocorr(z, max_lag=20):
    """Normalized autocorrelation of a (T, n) whitened sequence, averaged over components."""
    z = z - z.mean(0)
    T = len(z)
    den = np.sum(z * z, axis=0)
    r = np.array([np.sum(z[l:] * z[:T - l], axis=0) / den for l in range(1, max_lag + 1)])
    return r.mean(1), 1.96 / np.sqrt(T * z.shape[1])


def ljung_box(z, max_lag=20):
    """Ljung–Box Q per component and its χ²(max_lag) p-value."""
    z = z - z.mean(0)
    T = len(z)
    den = np.sum(z * z, axis=0)
    r = np.array([np.sum(z[l:] * z[:T - l], axis=0) / den for l in range(1, max_lag + 1)])
    Q = T * (T + 2) * np.sum(r**2 / (T - np.arange(1, max_lag + 1))[:, None], axis=0)
    return Q, chi2.sf(Q, max_lag)


def coverage(xs, m, P, level=0.95):
    """Fraction of per-component truth values inside the central marginal interval."""
    from scipy.stats import norm
    z = norm.ppf(0.5 + level / 2)
    s = np.sqrt(np.maximum(np.diagonal(P[1:], axis1=1, axis2=2), 1e-300))
    return float(np.nanmean(np.abs(xs[1:] - m[1:]) <= z * s))


def truth_columns(xs):
    """States with ground truth (columns of xs[1:] without NaNs); xs may be None."""
    if xs is None:
        return []
    return [j for j in range(xs.shape[1]) if np.all(np.isfinite(xs[1:, j]))]


def report(xs, res, alpha=0.05):
    """Scalar summary for one run of a filter/smoother result. xs may be None (measured data
    without ground truth: only the innovation statistics) or have NaN columns (truth for some
    states only: RMSE / NEES / coverage over those states)."""
    nx = res.m.shape[1]
    ok = bool(np.all(np.isfinite(res.m)))
    if not ok:
        return {"rmse": float("nan"), "rmse_per_state": [float("nan")] * nx, "diverged": True}
    cols = truth_columns(xs)
    out = {"rmse": float("nan"), "rmse_per_state": [float("nan")] * nx, "diverged": False,
           "truth_states": cols}
    if cols:
        X, M, P = xs[:, cols], res.m[:, cols], res.P[:, cols][:, :, cols]
        per = np.full(nx, np.nan); per[cols] = rmse(X, M, axis=0)
        out.update(rmse=rmse(X, M), rmse_per_state=per.tolist())
        nx = len(cols)
        xs, res_m, res_P = X, M, P
        e = nees(xs, res_m, res_P)
        T = len(e)
        lo, hi = chi2_band(nx, T, alpha)
        out.update(anees=float(np.mean(e)), anees_band=(lo, hi),
                   nees_frac_in=float(np.mean((e >= chi2.ppf(alpha / 2, nx)) &
                                              (e <= chi2.ppf(1 - alpha / 2, nx)))),
                   coverage95=coverage(xs, res_m, res_P))
    v, S = getattr(res, "v", None), getattr(res, "S", None)
    if ok and v is not None and S is not None and np.all(np.isfinite(v)):
        q = nis(v, S)
        ny = v.shape[1]
        out.update(anis=float(np.mean(q)), anis_band=chi2_band(ny, len(q), alpha))
        Qlb, p = ljung_box(whiten(v, S), min(20, len(v) // 4))
        out.update(ljung_box_p=p.tolist())
    return out
