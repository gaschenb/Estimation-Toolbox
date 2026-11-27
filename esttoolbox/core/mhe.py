"""Moving-horizon estimation and the batch MAP (Gauss–Newton) smoother.

At time k with window start s = max(0, k − N) the MHE solves

  min_{x_s..x_k}  ‖x_s − x̄_s‖²_{Π_s⁻¹} + Σ_{i=s+1}^{k} ‖x_i − f(x_{i−1}, i)‖²_{Q⁻¹}
                  + Σ_{i=max(s,1)}^{k} ‖y_i − h(x_i)‖²_{R⁻¹}        s.t.  lo ≤ x_i ≤ hi

by projected Levenberg–Marquardt on the sparse block-banded normal equations, with
analytic (or central-difference) block Jacobians. The arrival cost (x̄_s, Π_s) is the prior of x_s given y_{1:s−1}; when
the window slides, (x̄, Π) is advanced by one EKF step (update with y_s, predict to s+1)
linearized at the MHE estimate of x_s: the filtering-update arrival cost, which does not
double-count window measurements (Rao, Rawlings & Lee 2001).
P_k is the last diagonal block of the Gauss–Newton inverse Hessian.

batch_map(model, ys) is the full-horizon problem (s = 0, k = T): the MAP trajectory
of the whole state sequence, with Laplace (Gauss–Newton) marginal covariances.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .gaussian import FilterResult
from .linalg import chol, solve_psd, sym
from .smoothers import SmootherResult


class _Window:
    def __init__(self, model, ys, s, k, xbar, Pi):
        self.md, self.ys, self.s, self.k = model, ys, s, k
        self.n = k - s + 1
        self.xbar = xbar
        self.WPi = np.linalg.inv(chol(Pi))
        self.WQ = np.linalg.inv(chol(model.Q))
        self.WR = np.linalg.inv(chol(model.R))
        self.meas = [i for i in range(s, k + 1) if i >= 1]
        self.kf = np.arange(s + 1, k + 1)              # step index of each transition
        self.km = np.array(self.meas, dtype=int)
        self.mi = self.km - s                           # window row of each measurement
        self.Y = ys[self.km - 1] if self.meas else None
        nx, ny, n = model.nx, model.ny, self.n
        # static sparsity pattern of the whitened Jacobian
        rows, cols = [], []
        bi, bj = np.meshgrid(np.arange(nx), np.arange(nx), indexing="ij")
        rows.append(bi.ravel()); cols.append(bj.ravel())
        for j in range(1, n):
            r0 = nx * j
            rows += [r0 + bi.ravel(), r0 + bi.ravel()]
            cols += [j * nx + bj.ravel(), (j - 1) * nx + bj.ravel()]
        r0 = nx * n
        mi_, mj_ = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        for t, j in enumerate(self.mi):
            rows.append(r0 + t * ny + mi_.ravel()); cols.append(j * nx + mj_.ravel())
        self._rows, self._cols = np.concatenate(rows), np.concatenate(cols)
        self._shape = (nx * n + ny * len(self.meas), nx * n)

    def residual(self, z):
        md, nx = self.md, self.md.nx
        X = z.reshape(self.n, nx)
        r = [self.WPi @ (X[0] - self.xbar)]
        if self.n > 1:
            FX = md.f(X[:-1], self.kf[:, None])
            r.append(((X[1:] - FX) @ self.WQ.T).ravel())
        if self.meas:
            HX = md.h(X[self.mi], self.km[:, None])
            r.append((md.residual_y(self.Y, HX) @ self.WR.T).ravel())
        return np.concatenate(r)

    def jacobian(self, z):
        md, nx, n = self.md, self.md.nx, self.n
        X = z.reshape(n, nx)
        vals = [self.WPi.ravel()]
        if n > 1:
            F = md.F_batch(X[:-1], self.kf)                       # (n-1, nx, nx)
            WF = -np.einsum("ij,njk->nik", self.WQ, F)
            WQ = np.broadcast_to(self.WQ, F.shape)
            vals.append(np.stack([WQ, WF], axis=1).reshape(-1))
        if self.meas:
            H = md.Hk_batch(X[self.mi], self.km)                  # (m, ny, nx)
            vals.append((-np.einsum("ij,njk->nik", self.WR, H)).ravel())
        return sparse.csr_matrix((np.concatenate(vals), (self._rows, self._cols)),
                                 shape=self._shape)


def _solve(win, z0, lo, hi, max_iter=50, tol=1e-10):
    """Projected Levenberg–Marquardt on the sparse block-banded problem.
    Normal equations JᵀJ + λ diag(JᵀJ) are solved by sparse LU (O(n·nx³) per step)."""
    n = win.n
    LO = HI = None
    if lo is not None:
        LO, HI = np.tile(lo, n), np.tile(hi, n)
        z0 = np.clip(z0, LO, HI)
    z = np.asarray(z0, float).copy()
    r = win.residual(z)
    cost = float(r @ r)
    lam = 1e-4
    nfev = 1
    for _ in range(max_iter):
        J = win.jacobian(z)
        A = (J.T @ J).tocsc()
        g = J.T @ r
        d = A.diagonal() + 1e-12
        improved = False
        while lam < 1e12:
            dz = spsolve(A + sparse.diags(lam * d, format="csc"), -g)
            zn = z + dz
            if LO is not None:
                zn = np.clip(zn, LO, HI)
            rn = win.residual(zn)
            nfev += 1
            cn = float(rn @ rn)
            if np.isfinite(cn) and cn < cost:
                improved = True
                break
            lam *= 5.0
        if not improved:
            break
        step = np.linalg.norm(zn - z)
        rel = (cost - cn) / max(cost, 1e-300)
        z, r, cost = zn, rn, cn
        lam = max(lam / 5.0, 1e-12)
        if step <= tol * (1 + np.linalg.norm(z)) or rel < tol:
            break
    return z, cost, nfev, win.jacobian(z)


def _marginal_blocks(J, n, nx, which=None):
    Hs = (J.T @ J).toarray()
    Hinv = np.linalg.inv(Hs + 1e-12 * np.eye(len(Hs)) * max(np.trace(Hs) / len(Hs), 1.0))
    if which is not None:
        return sym(Hinv[which * nx:(which + 1) * nx, which * nx:(which + 1) * nx])
    return np.stack([sym(Hinv[j * nx:(j + 1) * nx, j * nx:(j + 1) * nx]) for j in range(n)])


def run_mhe(model, ys, horizon=10, constrained=True, max_iter=20):
    if not model.gaussian_obs:
        raise ValueError("MHE needs an additive Gaussian measurement model")
    T, nx, ny = len(ys), model.nx, model.ny
    lo = hi = None
    if constrained and model.state_bounds is not None:
        lo, hi = (np.asarray(b, float) for b in model.state_bounds)
    m = np.zeros((T + 1, nx)); P = np.zeros((T + 1, nx, nx))
    m[0], P[0] = model.m0, model.P0
    v = np.zeros((T, ny)); S = np.zeros((T, ny, ny))
    xbar, Pi, s = model.m0.copy(), model.P0.copy(), 0
    Z = model.m0[None].copy()                    # current window estimate (n, nx)
    t0 = time.perf_counter()
    for k in range(1, T + 1):
        # slide the window: drop x_s, propagate the arrival cost one step
        if k - s > horizon:
            xh = Z[0]                      # MHE estimate of x_s, linearization point
            xu, Pu = xbar, Pi
            if s >= 1:   # EKF measurement update of the arrival prior with y_s, linearized at xh
                H = model.H(xh, s)
                Sx = sym(H @ Pi @ H.T + model.R)
                K = solve_psd(Sx, (Pi @ H.T).T).T
                yl = model.h(xh[None], s)[0] + H @ (xbar - xh)
                xu = xbar + K @ model.residual_y(ys[s - 1], yl)
                I_KH = np.eye(nx) - K @ H
                Pu = sym(I_KH @ Pi @ I_KH.T + K @ model.R @ K.T)
            F = model.F(xh, s + 1)
            xbar = model.f(xh[None], s + 1)[0] + F @ (xu - xh)
            Pi = sym(F @ Pu @ F.T + model.Q)
            s += 1
            Z = Z[1:]
        # one-step prediction warm start + predictive innovation stats
        xp = model.f(Z[-1][None], k)[0]
        Fk = model.F(Z[-1], k)
        Pp = sym(Fk @ P[k - 1] @ Fk.T + model.Q)
        Hk = model.H(xp, k)
        v[k - 1] = model.residual_y(ys[k - 1], model.h(xp[None], k)[0])
        S[k - 1] = sym(Hk @ Pp @ Hk.T + model.R)
        Z = np.vstack([Z, xp])
        win = _Window(model, ys, s, k, xbar, Pi)
        z, _, _, J = _solve(win, Z.ravel(), lo, hi, max_iter)
        Z = z.reshape(win.n, nx)
        m[k] = Z[-1]
        P[k] = _marginal_blocks(J, win.n, nx, which=win.n - 1)
    nan = np.full_like(P, np.nan)
    return FilterResult("mhe", m, P, m.copy(), nan, nan, v, S, np.nan, time.perf_counter() - t0,
                        extras={"horizon": horizon, "constrained": lo is not None})


def batch_map(model, ys, init=None, constrained=True, max_iter=100):
    """Full-horizon MAP trajectory (nonlinear Gauss–Newton smoother)."""
    T, nx = len(ys), model.nx
    lo = hi = None
    if constrained and model.state_bounds is not None:
        lo, hi = (np.asarray(b, float) for b in model.state_bounds)
    if init is None:   # EKF/UKF filtered means are a good initial trajectory
        from .gaussian import make_filter
        try:
            init = make_filter("ukf", model).run(ys).m
        except Exception:
            init = None
        if init is None or not np.all(np.isfinite(init)):
            init = np.repeat(model.m0[None], T + 1, axis=0)
    t0 = time.perf_counter()
    win = _Window(model, ys, 0, T, model.m0, model.P0)
    z, cost, nfev, J = _solve(win, np.asarray(init, float).ravel(), lo, hi, max_iter)
    ms = z.reshape(T + 1, nx)
    Ps = _marginal_blocks(J, T + 1, nx)
    return SmootherResult("batch-map", ms, Ps, time.perf_counter() - t0,
                          extras={"cost": 0.5 * cost, "nfev": nfev})
