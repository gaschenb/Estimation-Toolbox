"""Parameter estimation for static problems y = f(θ) + ε, ε ~ N(0, σ²I).

Point estimates / asymptotics
  least-squares  bounded TRF with batched central-difference Jacobians, multi-start;
                 cov = s²(JᵀJ)⁻¹, s² = RSS/(n − p)
  mle            joint (θ, σ) maximum likelihood; observed-information covariance
  map            Gaussian prior, Laplace approximation at the mode; log-evidence
  fisher         Cramér–Rao bound at the LS estimate, FIM spectrum, condition number,
                 parameter correlations (local identifiability)
  profile        profile-likelihood 95 % intervals: {θ_j : 2[ℓ̂ − ℓ_p(θ_j)] ≤ χ²₁(0.95)}
Resampling
  bootstrap      residual / parametric / wild (Rademacher) bootstrap, percentile intervals
Bayesian (posterior ∝ N(y; f(θ), σ²I)·Π_j N(θ_j; μ_j, s_j²)·1[lo ≤ θ ≤ hi])
  rwm, am, ensemble, hmc   (see core.mcmc), multi-chain with R̂ and bulk/tail ESS
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares, brentq
from scipy.stats import chi2

from . import mcmc as _mc
from .linalg import sym


@dataclass
class Result:
    method: str
    theta_hat: np.ndarray
    stderr: np.ndarray
    cov: np.ndarray | None = None
    samples: np.ndarray | None = None
    sigma_hat: float | None = None
    ci: np.ndarray | None = None          # (p, 2) interval (90 % for samplers/bootstrap,
    ci_level: float | None = None         #  95 % for profile likelihood)
    runtime: float = 0.0
    extras: dict = field(default_factory=dict)


def jacobian(prob, theta, rel=1e-6):
    """Central differences, one batched forward call with 2p rows."""
    th = np.asarray(theta, float)
    p = th.size
    h = rel * np.maximum(np.abs(th), 1.0)
    # keep the stencil inside the box
    h = np.minimum(h, np.maximum((prob.hi - prob.lo) * 1e-3, 1e-12))
    E = np.diag(h)
    Tp = np.clip(th + E, prob.lo, prob.hi)
    Tm = np.clip(th - E, prob.lo, prob.hi)
    Y = prob.forward(np.vstack([Tp, Tm]))
    return ((Y[:p] - Y[p:]) / (np.diag(Tp) - np.diag(Tm))[:, None]).T


def _ls(prob, y, theta0=None, n_starts=1, rng=None, weights_prior=None, sigma=None):
    rng = np.random.default_rng(rng)
    starts = [np.clip(prob.prior_mean if theta0 is None else theta0, prob.lo, prob.hi)]
    for _ in range(n_starts - 1):
        starts.append(np.clip(prob.prior_mean + prob.prior_std * rng.standard_normal(prob.dim),
                              prob.lo, prob.hi))

    def res(th):
        r = prob.forward(th) - y
        if weights_prior is not None:
            r = np.concatenate([r / sigma, (th - prob.prior_mean) / prob.prior_std])
        return r

    def jac(th):
        J = jacobian(prob, th)
        if weights_prior is not None:
            J = np.vstack([J / sigma, np.diag(1.0 / prob.prior_std)])
        return J

    best = None
    for s0 in starts:
        sol = least_squares(res, s0, jac=jac, bounds=(prob.lo, prob.hi), method="trf",
                            x_scale="jac", xtol=1e-10, ftol=1e-10, gtol=1e-10)
        if best is None or sol.cost < best.cost:
            best = sol
    return best


def fit_least_squares(prob, y, sigma, n_starts=4, rng=None, **_):
    t0 = time.perf_counter()
    sol = _ls(prob, y, n_starts=n_starts, rng=rng)
    n, p = y.size, prob.dim
    rss = 2 * sol.cost
    s2 = rss / max(n - p, 1)
    J = jacobian(prob, sol.x)
    cov = s2 * np.linalg.pinv(J.T @ J)
    return Result("least-squares", sol.x, np.sqrt(np.diag(cov)), cov,
                  sigma_hat=float(np.sqrt(s2)), runtime=time.perf_counter() - t0,
                  extras={"rss": rss, "nfev": int(sol.nfev)})


def fit_mle(prob, y, sigma, rng=None, **_):
    """Joint ML in (θ, σ): θ̂ = θ̂_LS, σ̂² = RSS/n; covariance from the observed
    information of the profile, which is block-diagonal at the optimum."""
    t0 = time.perf_counter()
    sol = _ls(prob, y, n_starts=4, rng=rng)
    n = y.size
    s2 = 2 * sol.cost / n
    J = jacobian(prob, sol.x)
    cov = s2 * np.linalg.pinv(J.T @ J)
    ll = -0.5 * n * (np.log(2 * np.pi * s2) + 1.0)
    p = prob.dim + 1
    return Result("mle", sol.x, np.sqrt(np.diag(cov)), cov, sigma_hat=float(np.sqrt(s2)),
                  runtime=time.perf_counter() - t0,
                  extras={"loglik": ll, "sigma_stderr": float(np.sqrt(s2 / (2 * n))),
                          "aic": 2 * p - 2 * ll, "bic": p * np.log(n) - 2 * ll})


def fit_map(prob, y, sigma, **_):
    t0 = time.perf_counter()
    sol = _ls(prob, y, weights_prior=True, sigma=sigma)
    J = jacobian(prob, sol.x) / sigma
    H = J.T @ J + np.diag(1.0 / prob.prior_std**2)
    cov = np.linalg.inv(H)
    n, p = y.size, prob.dim
    # Laplace evidence: log p(y) ≈ log p(y|θ*) + log p(θ*) + p/2 log 2π − ½ log|H|
    r = prob.forward(sol.x) - y
    loglik = -0.5 * (r @ r) / sigma**2 - 0.5 * n * np.log(2 * np.pi * sigma**2)
    z = (sol.x - prob.prior_mean) / prob.prior_std
    logprior = -0.5 * z @ z - np.sum(np.log(prob.prior_std)) - 0.5 * p * np.log(2 * np.pi)
    logev = loglik + logprior + 0.5 * p * np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(H)[1]
    return Result("map", sol.x, np.sqrt(np.diag(cov)), cov, runtime=time.perf_counter() - t0,
                  extras={"log_evidence_laplace": float(logev)})


def fit_fisher(prob, y, sigma, rng=None, **_):
    t0 = time.perf_counter()
    sol = _ls(prob, y, n_starts=4, rng=rng)
    J = jacobian(prob, sol.x)
    fim = J.T @ J / sigma**2
    crb = np.linalg.pinv(fim)
    ev = np.linalg.eigvalsh(fim)
    sd = np.sqrt(np.diag(crb))
    corr = crb / np.outer(sd, sd)
    return Result("fisher", sol.x, sd, crb, runtime=time.perf_counter() - t0,
                  extras={"fim": fim, "fim_eigvals": ev,
                          "fim_condition": float(ev[-1] / max(ev[0], 1e-300)),
                          "correlation": corr})


def fit_profile(prob, y, sigma, level=0.95, rng=None, **_):
    """Profile-likelihood intervals with known σ (Wilks: χ²₁)."""
    t0 = time.perf_counter()
    sol = _ls(prob, y, n_starts=4, rng=rng)
    th = sol.x
    nll0 = sol.cost / sigma**2           # = ½ RSS / σ²
    crit = 0.5 * chi2.ppf(level, 1)
    ls_cov = sigma**2 * np.linalg.pinv(jacobian(prob, th).T @ jacobian(prob, th))
    se = np.sqrt(np.diag(ls_cov))
    ci = np.zeros((prob.dim, 2))
    profiles = {}
    for j in range(prob.dim):
        free = [i for i in range(prob.dim) if i != j]
        warm = {"x": th[free].copy()}

        def prof(v):
            if not free:
                r = prob.forward(np.array([v])) - y
                return 0.5 * (r @ r) / sigma**2 - nll0
            def res(u):
                full = th.copy(); full[free] = u; full[j] = v
                return prob.forward(full) - y
            def jac(u):
                full = th.copy(); full[free] = u; full[j] = v
                return jacobian(prob, full)[:, free]
            s = least_squares(res, np.clip(warm["x"], prob.lo[free], prob.hi[free]), jac=jac,
                              bounds=(prob.lo[free], prob.hi[free]), method="trf",
                              x_scale="jac", xtol=1e-10, ftol=1e-10)
            warm["x"] = s.x
            return s.cost / sigma**2 - nll0

        grid, vals = [], []
        for side, sgn in enumerate((-1, 1)):
            warm["x"] = th[free].copy()
            step = max(se[j], 1e-8 * max(abs(th[j]), 1))
            a, fa = th[j], 0.0
            bound = prob.lo[j] if sgn < 0 else prob.hi[j]
            found = None
            for _ in range(40):
                b = th[j] + sgn * step
                if (b - bound) * sgn >= 0:
                    b = bound
                fb = prof(b)
                grid.append(b); vals.append(fb)
                if fb >= crit:
                    found = (a, b)
                    break
                if b == bound:
                    break
                a, fa = b, fb
                step *= 1.6
            if found is None:
                ci[j, side] = bound          # interval runs into the box constraint
            else:
                ci[j, side] = brentq(lambda v: prof(v) - crit, *sorted(found), xtol=1e-10 *
                                     max(abs(th[j]), 1))
        o = np.argsort(grid)
        profiles[prob.param_names[j]] = (np.array(grid)[o], np.array(vals)[o])
    return Result("profile", th, (ci[:, 1] - ci[:, 0]) / (2 * 1.959964), ci=ci,
                  ci_level=level, runtime=time.perf_counter() - t0,
                  extras={"profiles": profiles, "crit": crit})


def fit_bootstrap(prob, y, sigma, n_boot=300, kind="residual", rng=None, **_):
    rng = np.random.default_rng(rng)
    t0 = time.perf_counter()
    sol = _ls(prob, y, n_starts=4, rng=rng)
    yhat = prob.forward(sol.x)
    res = y - yhat
    n, p = y.size, prob.dim
    res_c = (res - res.mean()) * np.sqrt(n / max(n - p, 1))     # centred, leverage-inflated
    s_hat = np.sqrt(np.sum(res**2) / max(n - p, 1))
    boots = np.zeros((n_boot, p))
    for b in range(n_boot):
        if kind == "residual":
            yb = yhat + rng.choice(res_c, size=n, replace=True)
        elif kind == "parametric":
            yb = yhat + s_hat * rng.standard_normal(n)
        elif kind == "wild":
            yb = yhat + res_c * rng.choice([-1.0, 1.0], size=n)
        else:
            raise ValueError(kind)
        boots[b] = _ls(prob, yb, theta0=sol.x).x
    ci = np.quantile(boots, [0.05, 0.95], axis=0).T
    return Result("bootstrap", sol.x, boots.std(0, ddof=1), np.cov(boots.T), samples=boots,
                  ci=ci, ci_level=0.90, runtime=time.perf_counter() - t0,
                  extras={"kind": kind, "n_boot": n_boot})


# ------------------------------ Bayesian -------------------------------------------
def log_posterior(prob, y, sigma):
    lo, hi, pm, ps = prob.lo, prob.hi, prob.prior_mean, prob.prior_std

    def logp_batch(Th):
        Th = np.atleast_2d(Th)
        inside = np.all((Th >= lo) & (Th <= hi), axis=1)
        out = np.full(len(Th), -np.inf)
        if inside.any():
            T_in = Th[inside]
            R = prob.forward(T_in) - y
            z = (T_in - pm) / ps
            out[inside] = -0.5 * np.sum(R * R, 1) / sigma**2 - 0.5 * np.sum(z * z, 1)
        return out

    def grad_batch(Th):
        """log posterior and gradient for all rows; the 2p-point Jacobian stencils of
        every row go through one batched forward call."""
        Th = np.atleast_2d(Th)
        lp = logp_batch(Th)
        G = np.zeros_like(Th)
        ok = np.nonzero(np.isfinite(lp))[0]
        if ok.size:
            T_ok = Th[ok]
            p = Th.shape[1]
            h = 1e-6 * np.maximum(np.abs(T_ok), 1.0)
            h = np.minimum(h, np.maximum((hi - lo) * 1e-3, 1e-12))
            Tp = np.clip(T_ok[:, None, :] + h[:, None, :] * np.eye(p), lo, hi)
            Tm = np.clip(T_ok[:, None, :] - h[:, None, :] * np.eye(p), lo, hi)
            Y = prob.forward(np.concatenate([Tp, Tm, T_ok[:, None, :]], axis=1)
                             .reshape(-1, p)).reshape(len(ok), 2 * p + 1, -1)
            dh = np.einsum("bii->bi", Tp - Tm)                       # (b, p)
            J = (Y[:, :p] - Y[:, p:2 * p]) / dh[:, :, None]          # (b, p, n) = Jᵀ
            r = Y[:, -1] - y
            G[ok] = -np.einsum("bpn,bn->bp", J, r) / sigma**2 - (T_ok - pm) / ps**2
        return lp, G

    return logp_batch, grad_batch


def _laplace_start(prob, y, sigma):
    sol = _ls(prob, y, weights_prior=True, sigma=sigma)
    J = jacobian(prob, sol.x) / sigma
    cov = np.linalg.inv(J.T @ J + np.diag(1.0 / prob.prior_std**2))
    return sol.x, sym(cov)


def _wrap(name, r, prob):
    s = r.summary()
    return Result(name, s["mean"], s["std"], np.cov(r.samples.T), samples=r.samples,
                  ci=np.column_stack([s["q05"], s["q95"]]), ci_level=0.90, runtime=r.runtime,
                  extras={"accept_rate": r.accept_rate, "rhat": s["rhat"],
                          "ess_bulk": s["ess_bulk"], "ess_tail": s["ess_tail"],
                          "ess_per_s": s["ess_per_s"], "n_evals": r.n_evals,
                          "chains": r.chains, **{k: v for k, v in r.extras.items()
                                                 if np.isscalar(v)}})


def fit_rwm(prob, y, sigma, n_iter=5000, burn=1000, n_chains=4, rng=None, **_):
    t0 = time.perf_counter()
    x0, cov = _laplace_start(prob, y, sigma)
    lp, _g = log_posterior(prob, y, sigma)
    r = _mc.rwm(lp, x0, cov, n_iter, burn, n_chains, rng)
    r.runtime = time.perf_counter() - t0
    return _wrap("rwm", r, prob)


def fit_am(prob, y, sigma, n_iter=5000, burn=1500, n_chains=4, rng=None, **_):
    t0 = time.perf_counter()
    x0, cov = _laplace_start(prob, y, sigma)
    lp, _g = log_posterior(prob, y, sigma)
    r = _mc.am(lp, x0, cov, n_iter, burn, n_chains, rng)
    r.runtime = time.perf_counter() - t0
    return _wrap("am", r, prob)


def fit_ensemble(prob, y, sigma, n_iter=2500, burn=500, rng=None, n_walkers=None, **_):
    t0 = time.perf_counter()
    x0, cov = _laplace_start(prob, y, sigma)
    lp, _g = log_posterior(prob, y, sigma)
    r = _mc.ensemble(lp, x0, cov, n_iter, burn, n_walkers, rng)
    r.runtime = time.perf_counter() - t0
    return _wrap("ensemble", r, prob)


def fit_hmc(prob, y, sigma, n_iter=1500, burn=500, n_chains=4, rng=None, **_):
    t0 = time.perf_counter()
    x0, cov = _laplace_start(prob, y, sigma)
    _lp, g = log_posterior(prob, y, sigma)
    r = _mc.hmc(g, x0, cov, n_iter, burn, n_chains, rng)
    r.runtime = time.perf_counter() - t0
    return _wrap("hmc", r, prob)


METHODS = {
    "least-squares": fit_least_squares,
    "mle": fit_mle,
    "map": fit_map,
    "fisher": fit_fisher,
    "profile": fit_profile,
    "bootstrap": fit_bootstrap,
    "rwm": fit_rwm,
    "am": fit_am,
    "ensemble": fit_ensemble,
    "hmc": fit_hmc,
}
