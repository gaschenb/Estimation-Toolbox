"""Gaussian assumed-density filters.

All filters share one loop: a predict step returning (m⁻, P⁻, C) with
C = Cov(x_{k-1}, x_k | y_{1:k-1}), and an update step returning the
innovation v and its covariance S. Storing C makes a single Gaussian RTS
smoother (smoothers.rts) valid for every filter here: RTS for the KF, ERTS for
EKF/IEKF, URTS / cubature RTS for the sigma-point filters, posterior-linearized
RTS for the IPLF and an ensemble RTS for EnKF/ETKF.

  kf     Kalman filter (Joseph-form update)
  srkf   square-root (array-form) Kalman filter, QR triangularization
  ekf    extended Kalman filter (analytic or central-difference Jacobians)
  iekf   iterated EKF (Gauss–Newton on the measurement update, MAP update)
  ukf    scaled unscented Kalman filter (α, β, κ)
  ckf    cubature Kalman filter (third-degree spherical-radial rule)
  ghkf   Gauss–Hermite Kalman filter (tensor rule of order p)
  iplf   iterated posterior-linearization filter (sigma-point SLR, García-Fernández 2015)
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

from .linalg import LOG2PI, chol, solve_psd, sym, tria


@dataclass
class FilterResult:
    method: str
    m: np.ndarray            # (T+1, nx) filtered means, m[0] = prior mean
    P: np.ndarray            # (T+1, nx, nx)
    mp: np.ndarray           # (T+1, nx) predicted means (mp[0] = m0)
    Pp: np.ndarray
    C: np.ndarray            # (T+1, nx, nx) Cov(x_{k-1}, x_k | y_{1:k-1}); C[0] unused
    v: np.ndarray            # (T, ny) innovations
    S: np.ndarray            # (T, ny, ny) innovation covariances
    loglik: float            # Σ log N(v_k; 0, S_k) (prediction-error decomposition)
    runtime: float
    diverged: bool = False
    extras: dict = field(default_factory=dict)

    @property
    def std(self):
        return np.sqrt(np.maximum(np.diagonal(self.P, axis1=1, axis2=2), 0.0))


# ---------------------------------------------------------------------------
# sigma-point rules: unit points ξ (n_pts, n) and weights for mean / covariance
# ---------------------------------------------------------------------------
def unscented_rule(n, alpha=1.0, beta=2.0, kappa=None):
    if kappa is None:
        kappa = max(3.0 - n, 0.0)
    lam = alpha**2 * (n + kappa) - n
    c = np.sqrt(n + lam)
    xi = np.vstack([np.zeros(n), c * np.eye(n), -c * np.eye(n)])
    wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
    wc = wm.copy()
    wm[0] = lam / (n + lam)
    wc[0] = lam / (n + lam) + (1 - alpha**2 + beta)
    return xi, wm, wc


def cubature_rule(n):
    xi = np.sqrt(n) * np.vstack([np.eye(n), -np.eye(n)])
    w = np.full(2 * n, 1.0 / (2 * n))
    return xi, w, w


def gauss_hermite_rule(n, order=3):
    z, w = hermegauss(order)          # probabilists' Hermite: weight e^{-x²/2}
    w = w / w.sum()
    pts = np.array(list(itertools.product(z, repeat=n)))
    ws = np.prod(np.array(list(itertools.product(w, repeat=n))), axis=1)
    return pts, ws, ws


RULES = {"ukf": unscented_rule, "ckf": cubature_rule, "ghkf": gauss_hermite_rule,
         "iplf": unscented_rule}


def _gauss_ll(v, S):
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        raise np.linalg.LinAlgError("innovation covariance not positive definite")
    return -0.5 * (v @ np.linalg.solve(S, v) + logdet + v.size * LOG2PI)


def _gain(P_xz, S):
    """P_xz S⁻¹ for small dense S (LU; S is symmetric positive definite)."""
    return np.linalg.solve(S, P_xz.T).T


# ---------------------------------------------------------------------------
class GaussianFilter:
    name = "gaussian"

    def __init__(self, model, **opts):
        if not model.gaussian_obs:
            raise ValueError(f"{self.name} needs an additive Gaussian measurement model; "
                             f"'{model.name}' has none (use a particle filter)")
        self.model, self.opts = model, opts

    # to be provided by subclasses
    def predict(self, m, P, k):
        raise NotImplementedError

    def update(self, m, P, y, k):
        raise NotImplementedError

    def run(self, ys):
        md = self.model
        T, nx, ny = len(ys), md.nx, md.ny
        m = np.zeros((T + 1, nx)); P = np.zeros((T + 1, nx, nx))
        mp = np.zeros_like(m); Pp = np.zeros_like(P); C = np.zeros_like(P)
        v = np.zeros((T, ny)); S = np.zeros((T, ny, ny))
        m[0] = mp[0] = md.m0; P[0] = Pp[0] = md.P0
        ll, diverged = 0.0, False
        t0 = time.perf_counter()
        err = np.seterr(all="ignore")
        for k in range(1, T + 1):
            try:
                mp[k], Pp[k], C[k] = self.predict(m[k - 1], P[k - 1], k)
                m[k], P[k], v[k - 1], S[k - 1] = self.update(mp[k], Pp[k], ys[k - 1], k)
                ll += _gauss_ll(v[k - 1], S[k - 1])
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                diverged = True
            if diverged or not (np.all(np.isfinite(m[k])) and np.all(np.isfinite(P[k]))):
                diverged = True
                m[k:] = np.nan; P[k:] = np.nan; mp[k:] = np.nan; Pp[k:] = np.nan
                v[k - 1:] = np.nan; S[k - 1:] = np.nan; ll = -np.inf
                break
        np.seterr(**err)
        return FilterResult(self.name, m, P, mp, Pp, C, v, S, ll,
                            time.perf_counter() - t0, diverged)

    # shared linear-Gaussian update for a (possibly linearized) measurement
    # y ≈ H x + b + e, e ~ N(0, Rl)
    def _linear_update(self, m, P, y, yhat, H, Rl):
        PHt = P @ H.T
        S = sym(H @ PHt + Rl)
        K = _gain(PHt, S)
        v = self.model.residual_y(y, yhat)
        I_KH = np.eye(len(m)) - K @ H
        Pn = sym(I_KH @ P @ I_KH.T + K @ Rl @ K.T)          # Joseph form
        return m + K @ v, Pn, v, S


class KF(GaussianFilter):
    name = "kf"

    def __init__(self, model, **opts):
        if not model.linear:
            raise ValueError("kf requires a linear model; use ekf/ukf")
        super().__init__(model, **opts)

    def predict(self, m, P, k):
        A = self.model.A
        PA = P @ A.T
        return A @ m, sym(A @ PA + self.model.Q), PA

    def update(self, m, P, y, k):
        C = self.model.C
        return self._linear_update(m, P, y, C @ m, C, self.model.R)


class SRKF(KF):
    """Array-form square-root KF: propagates Cholesky factors only."""
    name = "srkf"

    def run(self, ys):
        md = self.model
        A, Cm = md.A, md.C
        LQ, LR = chol(md.Q), chol(md.R)
        nx, ny, T = md.nx, md.ny, len(ys)
        m = np.zeros((T + 1, nx)); P = np.zeros((T + 1, nx, nx))
        mp = np.zeros_like(m); Pp = np.zeros_like(P); Cc = np.zeros_like(P)
        v = np.zeros((T, ny)); S = np.zeros((T, ny, ny))
        m[0] = mp[0] = md.m0; P[0] = Pp[0] = md.P0
        L = chol(md.P0)
        ll = 0.0
        t0 = time.perf_counter()
        for k in range(1, T + 1):
            Cc[k] = (L @ L.T) @ A.T
            mk = A @ m[k - 1]
            L = tria(np.hstack([A @ L, LQ]))
            mp[k], Pp[k] = mk, L @ L.T
            pre = np.zeros((ny + nx, ny + nx + ny))
            pre[:ny, :ny] = LR
            pre[:ny, ny:ny + nx] = Cm @ L
            pre[ny:, ny:ny + nx] = L
            post = tria(pre[:, :ny + nx])
            Ls = post[:ny, :ny]
            Kb = post[ny:, :ny]
            L = post[ny:, ny:ny + nx]
            vk = y = ys[k - 1] - Cm @ mk
            z = np.linalg.solve(Ls, y)
            m[k] = mk + Kb @ z
            P[k] = L @ L.T
            v[k - 1], S[k - 1] = vk, Ls @ Ls.T
            ll += -0.5 * (z @ z) - np.sum(np.log(np.abs(np.diag(Ls)))) - 0.5 * ny * LOG2PI
        return FilterResult(self.name, m, P, mp, Pp, Cc, v, S, ll, time.perf_counter() - t0)


class EKF(GaussianFilter):
    name = "ekf"

    def predict(self, m, P, k):
        F = self.model.F(m, k)
        PF = P @ F.T
        return self.model.f(m[None], k)[0], sym(F @ PF + self.model.Q), PF

    def update(self, m, P, y, k):
        H = self.model.H(m, k)
        return self._linear_update(m, P, y, self.model.h(m[None], k)[0], H, self.model.R)


class IEKF(EKF):
    """Iterated EKF: Gauss–Newton iterations x_{i+1} = m + K_i(y - h(x_i) - H_i(m - x_i)),
    converging to the MAP estimate of the single-step update."""
    name = "iekf"

    def update(self, m, P, y, k):
        md = self.model
        n_iter = int(self.opts.get("iterations", 10))
        tol = float(self.opts.get("tol", 1e-8))
        x = m.copy()
        for _ in range(n_iter):
            H = md.H(x, k)
            yhat = md.h(x[None], k)[0] + H @ (m - x)
            xn, Pn, _, S = self._linear_update(m, P, y, yhat, H, md.R)
            done = np.linalg.norm(xn - x) <= tol * (1 + np.linalg.norm(x))
            x = xn
            if done:
                break
        H = md.H(x, k)
        Sx = sym(H @ P @ H.T + md.R)
        K = solve_psd(Sx, (P @ H.T).T).T
        I_KH = np.eye(len(m)) - K @ H
        Pn = sym(I_KH @ P @ I_KH.T + K @ md.R @ K.T)
        # innovation statistics at the prior mean (what NIS tests)
        Hm = md.H(m, k)
        v = md.residual_y(y, md.h(m[None], k)[0])
        return x, Pn, v, sym(Hm @ P @ Hm.T + md.R)


class SigmaPointKF(GaussianFilter):
    def __init__(self, model, kind="ukf", **opts):
        self.name = kind
        super().__init__(model, **opts)
        n = model.nx
        if kind == "ukf" or kind == "iplf":
            self.xi, self.wm, self.wc = unscented_rule(
                n, opts.get("alpha", 1.0), opts.get("beta", 2.0), opts.get("kappa", None))
        elif kind == "ckf":
            self.xi, self.wm, self.wc = cubature_rule(n)
        elif kind == "ghkf":
            self.xi, self.wm, self.wc = gauss_hermite_rule(n, int(opts.get("order", 3)))
        else:
            raise ValueError(kind)

    def points(self, m, P):
        return m + self.xi @ chol(P).T

    def predict(self, m, P, k):
        X = self.points(m, P)
        Y = self.model.f(X, k)
        mp = self.wm @ Y
        dY, dX = Y - mp, X - m
        Pp = sym((self.wc[:, None] * dY).T @ dY + self.model.Q)
        C = (self.wc[:, None] * dX).T @ dY
        return mp, Pp, C

    def _moments(self, m, P, k):
        md = self.model
        X = self.points(m, P)
        Z = md.h(X, k)
        dZ0 = md.residual_y(Z, Z[0])
        zbar = Z[0] + self.wm @ dZ0
        dZ = md.residual_y(Z, zbar)
        Pzz = (self.wc[:, None] * dZ).T @ dZ
        Pxz = (self.wc[:, None] * (X - m)).T @ dZ
        return zbar, Pzz, Pxz

    def update(self, m, P, y, k):
        md = self.model
        if self.name == "iplf":
            return self._iplf_update(m, P, y, k)
        zbar, Pzz, Pxz = self._moments(m, P, k)
        S = sym(Pzz + md.R)
        K = _gain(Pxz, S)
        v = md.residual_y(y, zbar)
        return m + K @ v, sym(P - K @ S @ K.T), v, S

    def _iplf_update(self, m, P, y, k):
        """Iterated statistical linear regression of h w.r.t. the current posterior."""
        md = self.model
        n_iter = int(self.opts.get("iterations", 5))
        mi, Pi = m, P
        for _ in range(n_iter):
            zbar, Pzz, Pxz = self._moments(mi, Pi, k)
            A = solve_psd(Pi, Pxz).T                # (ny, nx)
            b = zbar - A @ mi
            Om = sym(Pzz - A @ Pi @ A.T)
            mn, Pn, _, S = self._linear_update(m, P, y, A @ m + b, A, md.R + Om)
            step = np.linalg.norm(mn - mi)
            mi, Pi = mn, Pn
            if step < 1e-9 * (1 + np.linalg.norm(mi)):
                break
        # report the prior-predictive innovation (sigma points of the prior)
        zbar, Pzz, _ = self._moments(m, P, k)
        return mi, Pi, md.residual_y(y, zbar), sym(Pzz + md.R)


def make_filter(name, model, **opts):
    name = name.lower()
    if name == "kf":
        return KF(model, **opts)
    if name == "srkf":
        return SRKF(model, **opts)
    if name == "ekf":
        return EKF(model, **opts)
    if name == "iekf":
        return IEKF(model, **opts)
    if name in ("ukf", "ckf", "ghkf", "iplf"):
        return SigmaPointKF(model, kind=name, **opts)
    raise KeyError(name)
