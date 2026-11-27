"""System identification: parameters θ of a state-space family θ ↦ model(θ).

  pem     prediction-error / maximum-likelihood (optionally MAP) estimation: maximize
          ℓ(θ) = Σ_k log N(v_k; 0, S_k) from a Gaussian filter (KF exact for linear models,
          EKF/UKF/CKF approximate otherwise) over an unconstrained reparametrization u,
          θ = lo + (hi − lo)·sigmoid(u). Standard errors from the numerical Hessian in u,
          mapped to θ by the delta method.
  em      expectation–maximization for linear-Gaussian models (Shumway & Stoffer 1982):
          RTS smoother + lag-one cross-covariances in the E-step, closed-form M-step for A,
          C, Q, R, m0, P0 (full, scaled or fixed). ℓ is non-decreasing by construction.
  pmmh    particle marginal Metropolis–Hastings (Andrieu, Doucet & Holenstein 2010):
          pseudo-marginal MH on u using the unbiased bootstrap-PF likelihood, prior on u
          N(T(prior_mean), prior_std²), adaptive proposal during burn-in.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from .gaussian import make_filter
from .linalg import sym
from .particle import pf_loglik
from .smoothers import rts


class Transform:
    def __init__(self, lo, hi):
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)

    def to_theta(self, u):
        return self.lo + (self.hi - self.lo) * expit(u)

    def to_u(self, th):
        z = (np.asarray(th, float) - self.lo) / (self.hi - self.lo)
        return logit(np.clip(z, 1e-12, 1 - 1e-12))

    def dtheta_du(self, u):
        s = expit(u)
        return (self.hi - self.lo) * s * (1 - s)

    def log_jac(self, u):
        return float(np.sum(np.log(self.dtheta_du(u))))


@dataclass
class SysIdResult:
    method: str
    param_names: list
    theta_hat: np.ndarray
    stderr: np.ndarray
    cov: np.ndarray | None
    loglik: float
    runtime: float
    samples: np.ndarray | None = None
    ci: np.ndarray | None = None
    history: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)


def _hessian(fun, u, h=1e-3):
    n = u.size
    H = np.zeros((n, n))
    f0 = fun(u)
    for i in range(n):
        ei = np.zeros(n); ei[i] = h
        H[i, i] = (fun(u + ei) - 2 * f0 + fun(u - ei)) / h**2
        for j in range(i + 1, n):
            ej = np.zeros(n); ej[j] = h
            H[i, j] = H[j, i] = (fun(u + ei + ej) - fun(u + ei - ej) - fun(u - ei + ej)
                                 + fun(u - ei - ej)) / (4 * h * h)
    return H


def filter_loglik(problem, theta, ys, filt="ukf", **opts):
    try:
        md = problem.model(theta)
        return make_filter(filt, md, **opts).run(ys).loglik
    except (ValueError, np.linalg.LinAlgError):
        return -np.inf


def pem(problem, ys, filt=None, theta0=None, use_prior=False, free=None, maxiter=300):
    """free: indices of the parameters to estimate (others fixed at theta0)."""
    t0 = time.perf_counter()
    filt = filt or ("kf" if problem.model().linear else "ukf")
    tr = Transform(problem.lo, problem.hi)
    th0 = np.asarray(problem.prior_mean if theta0 is None else theta0, float).copy()
    free = np.arange(len(th0)) if free is None else np.asarray(free)
    u_prior = tr.to_u(problem.prior_mean)
    hist = []

    def full(uf):
        u = tr.to_u(th0); u[free] = uf
        return u

    def nlp(uf):
        u = full(uf)
        ll = filter_loglik(problem, tr.to_theta(u), ys, filt)
        if not np.isfinite(ll):
            return 1e12
        val = -ll
        if use_prior:
            z = (u[free] - u_prior[free]) / problem.prior_std[free]
            val += 0.5 * z @ z
        return val

    def cb(uf):
        hist.append(float(-nlp(uf)))

    u_init = tr.to_u(th0)[free]
    res = minimize(nlp, u_init, method="Nelder-Mead" if len(free) == 1 else "BFGS",
                   callback=cb, options={"maxiter": maxiter})
    if len(free) > 1 and not res.success:   # BFGS stalled (noisy FD gradients): polish
        res2 = minimize(nlp, res.x, method="Nelder-Mead",
                        options={"maxiter": 200 * len(free), "xatol": 1e-6, "fatol": 1e-8})
        if res2.fun < res.fun:
            res = res2
    uh = res.x
    H = _hessian(nlp, uh)
    try:
        cov_u = np.linalg.inv(sym(H))
        if np.any(np.diag(cov_u) < 0):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        cov_u = np.full((len(free), len(free)), np.nan)
    u_full = full(uh)
    th = tr.to_theta(u_full)
    Jd = tr.dtheta_du(u_full)[free]
    cov = np.full((len(th0), len(th0)), 0.0)
    cov[np.ix_(free, free)] = Jd[:, None] * cov_u * Jd[None, :]
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    ll = filter_loglik(problem, th, ys, filt)
    k = len(free)
    return SysIdResult("pem-" + filt + ("-map" if use_prior else ""), problem.param_names, th,
                       se, cov, ll, time.perf_counter() - t0, history=hist,
                       extras={"nfev": int(res.nfev), "aic": 2 * k - 2 * ll,
                               "bic": k * np.log(len(ys)) - 2 * ll, "free": free.tolist()})


def em_lgssm(model, ys, n_iter=50, estimate=("Q", "R"), Q_mode="scaled", R_mode="scaled",
             tol=1e-6):
    """EM for x_k = A x_{k−1} + q, y_k = C x_k + r.

    estimate ⊂ {A, C, Q, R, m0, P0}. Q_mode/R_mode ∈ {full, scaled, diag}: 'scaled' fits
    Q = s·Q_init (structure fixed, e.g. white-noise acceleration), 'diag' a diagonal.
    Returns the fitted model, the log-likelihood trace (non-decreasing up to round-off)."""
    from .ssm import LinearGaussianSSM
    t0 = time.perf_counter()
    A, Cm, Q, R = model.A.copy(), model.C.copy(), model.Q.copy(), model.R.copy()
    m0, P0 = model.m0.copy(), model.P0.copy()
    Qb, Rb = model.Q.copy(), model.R.copy()
    T = len(ys)
    lls = []
    md = model
    for it in range(n_iter):
        md = LinearGaussianSSM(model.name, A, Cm, Q, R, m0, P0, state_names=model.state_names,
                               obs_names=model.obs_names, dt=model.dt)
        fr = make_filter("kf", md).run(ys)
        lls.append(fr.loglik)
        if it > 0 and abs(lls[-1] - lls[-2]) < tol * abs(lls[-1]):
            break
        sm = rts(fr)
        ms, Ps, Pc = sm.m, sm.P, sm.Pcross
        Ex = Ps + np.einsum("ti,tj->tij", ms, ms)                # E[x_k x_kᵀ]
        Sig = Ex[1:].mean(0)                                        # (1/T) Σ_{k=1}^T
        Phi = Ex[:-1].mean(0)                                       # (1/T) Σ_{k=0}^{T−1}
        Cc = (Pc + np.einsum("ti,tj->tij", ms[1:], ms[:-1])).mean(0)   # E[x_k x_{k−1}ᵀ]
        Bm = np.einsum("ti,tj->ij", ys, ms[1:]) / T                 # E[y_k x_kᵀ]
        Dm = ys.T @ ys / T
        if "A" in estimate:
            A = Cc @ np.linalg.inv(Phi)
        if "C" in estimate:
            Cm = Bm @ np.linalg.inv(Sig)
        if "Q" in estimate:
            EQ = sym(Sig - Cc @ A.T - A @ Cc.T + A @ Phi @ A.T)
            if Q_mode == "full":
                Q = EQ
            elif Q_mode == "diag":
                Q = np.diag(np.diag(EQ))
            else:
                Q = Qb * np.trace(np.linalg.solve(Qb, EQ)) / len(Qb)
        if "R" in estimate:
            ER = sym(Dm - Cm @ Bm.T - Bm @ Cm.T + Cm @ Sig @ Cm.T)
            if R_mode == "full":
                R = ER
            elif R_mode == "diag":
                R = np.diag(np.diag(ER))
            else:
                R = Rb * np.trace(np.linalg.solve(Rb, ER)) / len(Rb)
        if "m0" in estimate:
            m0 = ms[0].copy()
        if "P0" in estimate:
            P0 = sym(Ps[0])
    return md, np.array(lls), time.perf_counter() - t0


def em(problem, ys, n_iter=100, **kw):
    """EM wrapper for the cv-tracking family θ = (q, r): Q = q·Q₁, R = r²I."""
    md0 = problem.model(problem.prior_mean)
    md, lls, rt = em_lgssm(md0, ys, n_iter=n_iter, **kw)
    q0, r0 = problem.prior_mean
    q = q0 * np.trace(np.linalg.solve(md0.Q, md.Q)) / md.nx
    r = np.sqrt(np.trace(md.R) / md.ny)
    th = np.array([q, r])
    return SysIdResult("em", problem.param_names, th, np.full(2, np.nan), None, lls[-1], rt,
                       history=lls.tolist(), extras={"Q": md.Q, "R": md.R,
                                                     "iterations": len(lls)})


def pmmh(problem, ys, n_particles=300, n_iter=3000, burn=1000, theta0=None, cov_u=None,
         rng=None, free=None, progress=None):
    """PMMH on u = T(θ). Proposal: adaptive Gaussian random walk, frozen after burn-in."""
    rng = np.random.default_rng(rng)
    t0 = time.perf_counter()
    tr = Transform(problem.lo, problem.hi)
    th0 = np.asarray(problem.prior_mean if theta0 is None else theta0, float)
    free = np.arange(len(th0)) if free is None else np.asarray(free)
    d = len(free)
    u_prior = tr.to_u(problem.prior_mean)
    u0 = tr.to_u(th0)

    def logpost(uf):
        u = u0.copy(); u[free] = uf
        th = tr.to_theta(u)
        try:
            ll = pf_loglik(problem.model(th), ys, n_particles, rng=rng)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return -np.inf, -np.inf
        z = (uf - u_prior[free]) / problem.prior_std[free]
        return ll - 0.5 * z @ z + tr.log_jac(u), ll     # prior on u: no Jacobian needed;
        #   log_jac converts to a flat-in-θ reference; kept for a proper θ-scale posterior

    Sig = np.diag((0.1 * problem.prior_std[free]) ** 2) if cov_u is None else cov_u
    logs = np.log(2.38**2 / d)
    x = u0[free].copy()
    lp, ll = logpost(x)
    mu, n_seen = x.copy(), 1
    keep = np.zeros((n_iter - burn, d)); lls = np.zeros(n_iter - burn)
    acc = 0
    for i in range(n_iter):
        L = np.exp(0.5 * logs) * np.linalg.cholesky(Sig + 1e-10 * np.eye(d))
        xp = x + L @ rng.standard_normal(d)
        lpp, llp = logpost(xp)
        a = min(0.0, lpp - lp) if np.isfinite(lpp) else -np.inf
        if np.log(rng.random()) < a:
            x, lp, ll = xp, lpp, llp
            if i >= burn:
                acc += 1
        if i < burn:
            g = 1.0 / (i + 1) ** 0.6
            logs += g * (np.exp(a) - 0.234)
            n_seen += 1
            w = 1.0 / n_seen
            dx = x - mu
            mu = mu + w * dx
            Sig = sym(Sig + w * (np.outer(dx, x - mu) - Sig)) if i > 50 else Sig
        else:
            keep[i - burn], lls[i - burn] = x, ll
        if progress is not None and i % 100 == 0:
            progress(i, n_iter)
    U = np.tile(u0, (len(keep), 1)); U[:, free] = keep
    TH = tr.to_theta(U)
    from .diagnostics import ess_bulk
    ess = np.array([ess_bulk(TH[None, :, j]) for j in free])
    ci = np.quantile(TH, [0.05, 0.95], axis=0).T
    return SysIdResult("pmmh", problem.param_names, TH.mean(0), TH.std(0, ddof=1),
                       np.cov(TH.T), float(np.mean(lls)), time.perf_counter() - t0,
                       samples=TH, ci=ci, history=lls.tolist(),
                       extras={"accept_rate": acc / (n_iter - burn), "ess_bulk": ess,
                               "n_particles": n_particles, "free": free.tolist()})
