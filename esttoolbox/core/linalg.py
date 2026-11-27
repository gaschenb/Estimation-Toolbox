"""Numerically careful dense linear algebra used throughout the toolbox."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, expm, qr, solve_triangular

LOG2PI = np.log(2.0 * np.pi)


def sym(P):
    """Symmetric part, on the last two axes."""
    return 0.5 * (P + np.swapaxes(P, -1, -2))


def chol(P, jitter=1e-12, max_tries=8):
    """Lower Cholesky factor of a (batch of) PSD matrix, escalating diagonal
    jitter relative to the mean diagonal if the factorization fails."""
    P = sym(np.asarray(P, float))
    try:
        return np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        pass
    n = P.shape[-1]
    scale = np.mean(np.abs(np.diagonal(P, axis1=-2, axis2=-1))) or 1.0
    eye = np.eye(n)
    j = jitter
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(P + j * scale * eye)
        except np.linalg.LinAlgError:
            j *= 100.0
    # Last resort: eigenvalue clipping.
    w, V = np.linalg.eigh(P)
    w = np.maximum(w, jitter * scale)
    return np.linalg.cholesky(sym((V * w[..., None, :]) @ np.swapaxes(V, -1, -2)))


def solve_psd(S, B):
    """S^{-1} B for symmetric positive definite S (Cholesky)."""
    try:
        c = cho_factor(sym(S), lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        c = (chol(S), True)
    return cho_solve(c, B, check_finite=False)


def gauss_logpdf(v, S):
    """log N(v; 0, S) and the Mahalanobis form v^T S^{-1} v."""
    L = chol(S)
    z = solve_triangular(L, v, lower=True, check_finite=False)
    maha = float(z @ z)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (maha + logdet + v.size * LOG2PI), maha


def batch_mahalanobis(E, P):
    """e_i^T P_i^{-1} e_i for E (..., n), P (..., n, n)."""
    L = chol(P)
    z = np.linalg.solve(L, E[..., None])[..., 0]
    return np.sum(z * z, axis=-1)


def tria(A):
    """Lower-triangular L with L L^T = A A^T (QR of A^T), signs normalized."""
    R = qr(A.T, mode="economic")[1]
    L = R.T
    s = np.sign(np.diag(L))
    s[s == 0] = 1.0
    return L * s


def van_loan(A, L, Qc, dt):
    """Exact discretization of dx = A x dt + L dβ, E[dβ dβ^T] = Qc dt.
    Returns (F, Q) with F = expm(A dt), Q = ∫ e^{As} L Qc L^T e^{A^T s} ds."""
    n = A.shape[0]
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -A
    M[:n, n:] = L @ Qc @ L.T
    M[n:, n:] = A.T
    E = expm(M * dt)
    F = E[n:, n:].T
    Q = F @ E[:n, n:]
    return F, sym(Q)


def fd_jacobian(fun, x, h=None):
    """Central-difference Jacobian of a vectorized map fun: (..., n) -> (..., m)
    at one point x (n,). All 2n perturbed points are evaluated in one call."""
    x = np.asarray(x, float)
    n = x.size
    if h is None:
        h = 1e-6 * np.maximum(np.abs(x), 1.0)
    E = np.diag(h)
    X = np.concatenate([x + E, x - E], axis=0)
    Y = fun(X)
    return ((Y[:n] - Y[n:]) / (2.0 * h[:, None])).T


def fd_jacobian_batch(fun, X, h=None):
    """Central-difference Jacobians at many points X (N, n) -> (N, m, n)."""
    X = np.asarray(X, float)
    N, n = X.shape
    if h is None:
        h = 1e-6 * np.maximum(np.abs(X), 1.0)  # (N, n)
    pts = np.repeat(X[:, None, :], 2 * n, axis=1)  # (N, 2n, n)
    idx = np.arange(n)
    pts[:, idx, idx] += h
    pts[:, n + idx, idx] -= h
    Y = fun(pts.reshape(N * 2 * n, n)).reshape(N, 2 * n, -1)
    D = (Y[:, :n] - Y[:, n:]) / (2.0 * h[:, :, None])  # (N, n, m)
    return np.swapaxes(D, 1, 2)
