"""State-space model interface.

Discrete-time model with states x_0..x_T and measurements y_1..y_T:

    x_0 ~ N(m0, P0)
    x_k = f(x_{k-1}, k) + q_k,      q_k ~ N(0, Q)
    y_k = h(x_k, k) + r_k,          r_k ~ N(0, R)

f and h are vectorized over leading axes: f(X, k) with X of shape (..., nx); k is a
step index, either a scalar or an array broadcastable against X[..., :1].
Models with a non-additive or non-Gaussian measurement (stochastic volatility)
override `obs_loglik` / `sample_obs` and set `gaussian_obs = False`; only
sampling-based methods (particle filters, PMMH) accept them.
"""
from __future__ import annotations

import numpy as np

from .linalg import chol, fd_jacobian, fd_jacobian_batch, LOG2PI


class StateSpaceModel:
    gaussian_obs = True
    linear = False

    def __init__(self, name, nx, ny, f, h, Q, R, m0, P0, *, F_jac=None, H_jac=None,
                 state_names=None, obs_names=None, dt=1.0, angle_obs=(),
                 x0=None, state_bounds=None, description=""):
        self.name, self.nx, self.ny = name, nx, ny
        self._f, self._h = f, h
        self._Fj, self._Hj = F_jac, H_jac
        self.Q = np.atleast_2d(np.asarray(Q, float))
        self.R = np.atleast_2d(np.asarray(R, float))
        self.m0 = np.asarray(m0, float)
        self.P0 = np.atleast_2d(np.asarray(P0, float))
        self.x0 = None if x0 is None else np.asarray(x0, float)
        self.state_names = state_names or [f"x{i}" for i in range(nx)]
        self.obs_names = obs_names or [f"y{i}" for i in range(ny)]
        self.dt = dt
        self.angle_obs = tuple(angle_obs)  # measurement components that are angles
        self.state_bounds = state_bounds   # (lo, hi) physical bounds, used by MHE
        self.description = description
        self._LQ = chol(self.Q)
        self._LR = chol(self.R)

    # ----- model functions -------------------------------------------------
    def f(self, X, k):
        return self._f(np.asarray(X, float), k)

    def h(self, X, k):
        return self._h(np.asarray(X, float), k)

    def F(self, x, k):
        if self._Fj is not None:
            return self._Fj(np.asarray(x, float), k)
        return fd_jacobian(lambda X: self.f(X, k), x)

    def H(self, x, k):
        if self._Hj is not None:
            return self._Hj(np.asarray(x, float), k)
        return fd_jacobian(lambda X: self.h(X, k), x)

    def H_batch(self, X, k):
        """Jacobians of h at many points (N, nx) -> (N, ny, nx), one vectorized call."""
        return fd_jacobian_batch(lambda Z: self.h(Z, k), X)

    def F_batch(self, X, ks):
        """Jacobians of f at X (N, nx) with per-row step indices ks (N,) -> (N, nx, nx)."""
        ks = np.asarray(ks)
        rep = np.repeat(ks, 2 * self.nx)[:, None]
        return fd_jacobian_batch(lambda Z: self.f(Z, rep), X)

    def Hk_batch(self, X, ks):
        ks = np.asarray(ks)
        rep = np.repeat(ks, 2 * self.nx)[:, None]
        return fd_jacobian_batch(lambda Z: self.h(Z, rep), X)

    def residual_y(self, a, b):
        """a - b with angle components wrapped to (-π, π]."""
        d = a - b
        if self.angle_obs:
            d = np.array(d, copy=True)
            i = list(self.angle_obs)
            d[..., i] = (d[..., i] + np.pi) % (2 * np.pi) - np.pi
        return d

    # ----- densities / sampling -------------------------------------------
    def obs_loglik(self, y, X, k):
        """log p(y | x) for each row of X (N, nx) -> (N,)."""
        d = self.residual_y(y, self.h(X, k))
        z = np.linalg.solve(self._LR, d.T).T
        return (-0.5 * np.sum(z * z, axis=1) - np.sum(np.log(np.diag(self._LR)))
                - 0.5 * self.ny * LOG2PI)

    def sample_obs(self, X, k, rng):
        Y = self.h(X, k)
        return Y + rng.standard_normal(Y.shape) @ self._LR.T

    def trans_logpdf(self, Xn, X, k):
        """log p(x_k = Xn | x_{k-1} = X), broadcasting over leading axes."""
        d = Xn - self.f(X, k)
        z = np.linalg.solve(self._LQ, np.moveaxis(d, -1, 0).reshape(self.nx, -1))
        z = z.reshape((self.nx,) + d.shape[:-1])
        return (-0.5 * np.sum(z * z, axis=0) - np.sum(np.log(np.diag(self._LQ)))
                - 0.5 * self.nx * LOG2PI)

    def sample_trans(self, X, k, rng):
        Xn = self.f(X, k)
        return Xn + rng.standard_normal(Xn.shape) @ self._LQ.T

    def sample_x0(self, rng, n=None):
        """Draws from N(m0, P0), truncated to state_bounds when the model has them."""
        L = chol(self.P0)
        if n is None:
            return self.sample_x0(rng, 1)[0]
        X = self.m0 + rng.standard_normal((n, self.nx)) @ L.T
        if self.state_bounds is not None:
            lo, hi = self.state_bounds
            for _ in range(100):
                bad = np.any((X < lo) | (X > hi), axis=1)
                if not bad.any():
                    break
                X[bad] = self.m0 + rng.standard_normal((bad.sum(), self.nx)) @ L.T
            X = np.clip(X, lo, hi)
        return X

    def simulate(self, T, rng=None, x0=None):
        """Ground truth xs (T+1, nx) and measurements ys (T, ny)."""
        rng = np.random.default_rng(rng)
        if x0 is None:
            x0 = self.x0 if self.x0 is not None else self.sample_x0(rng)
        xs = np.empty((T + 1, self.nx))
        ys = np.empty((T, self.ny))
        xs[0] = x0
        for k in range(1, T + 1):
            xs[k] = self.sample_trans(xs[k - 1][None], k, rng)[0]
            ys[k - 1] = self.sample_obs(xs[k][None], k, rng)[0]
        return xs, ys


class LinearGaussianSSM(StateSpaceModel):
    """x_k = A x_{k-1} + q,  y_k = C x_k + r (time invariant)."""
    linear = True

    def __init__(self, name, A, C, Q, R, m0, P0, **kw):
        self.A = np.atleast_2d(np.asarray(A, float))
        self.C = np.atleast_2d(np.asarray(C, float))
        A_, C_ = self.A, self.C
        super().__init__(name, A_.shape[0], C_.shape[0],
                         lambda X, k: X @ A_.T, lambda X, k: X @ C_.T, Q, R, m0, P0,
                         F_jac=lambda x, k: A_, H_jac=lambda x, k: C_, **kw)

    def H_batch(self, X, k):
        return np.broadcast_to(self.C, (len(X),) + self.C.shape)
