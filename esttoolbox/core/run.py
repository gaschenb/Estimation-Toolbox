"""Uniform entry points: run any state estimator by name, smooth it, Monte Carlo it."""
from __future__ import annotations

import time

import numpy as np
from scipy.stats import chi2

from . import consistency as cs
from .ensemble import run_enkf
from .gaussian import make_filter
from .mhe import batch_map, run_mhe
from .particle import run_pf
from .smoothers import ffbsi, rts

STATE_METHODS = {
    "kf": "Kalman filter (Joseph form)",
    "srkf": "square-root (array) Kalman filter",
    "ekf": "extended Kalman filter",
    "iekf": "iterated EKF (Gauss–Newton MAP update)",
    "ukf": "unscented Kalman filter (α, β, κ)",
    "ckf": "cubature Kalman filter",
    "ghkf": "Gauss–Hermite Kalman filter",
    "iplf": "iterated posterior-linearization filter",
    "enkf": "stochastic ensemble Kalman filter",
    "etkf": "ensemble transform Kalman filter",
    "pf": "bootstrap particle filter",
    "rpf": "regularized particle filter (kernel jitter)",
    "pf-ekf": "particle filter, linearized optimal proposal",
    "mhe": "moving-horizon estimation (constrained)",
}
GAUSSIAN = ("kf", "srkf", "ekf", "iekf", "ukf", "ckf", "ghkf", "iplf")


def applicable(method, model):
    if method in ("kf", "srkf"):
        return model.linear
    if method in ("pf", "rpf"):
        return True
    if method == "ghkf" and model.nx > 6:       # tensor rule: 3^nx points
        return False
    return model.gaussian_obs


def run_method(method, model, ys, rng=None, **opts):
    rng = np.random.default_rng(rng)
    if method in GAUSSIAN:
        keys = {"alpha", "beta", "kappa", "order", "iterations"}
        return make_filter(method, model, **{k: v for k, v in opts.items() if k in keys}).run(ys)
    if method in ("enkf", "etkf"):
        return run_enkf(model, ys, opts.get("n_ens", 100), method, opts.get("inflation", 1.0), rng)
    if method in ("pf", "rpf", "pf-ekf"):
        return run_pf(model, ys, opts.get("n_particles", 1000),
                      "ekf" if method == "pf-ekf" else "bootstrap",
                      opts.get("resampling", "systematic"), opts.get("ess_threshold", 0.5), rng,
                      regularize=method == "rpf")
    if method == "mhe":
        return run_mhe(model, ys, opts.get("horizon", 10), opts.get("constrained", True))
    if method == "batch-map":
        return batch_map(model, ys)
    raise KeyError(f"unknown method '{method}'")


def smooth(result, model, rng=None, n_traj=100):
    """RTS-type smoother for Gaussian/ensemble filters, FFBSi for particle filters."""
    if hasattr(result, "particles"):
        if result.diverged:
            return None
        return ffbsi(result, model, n_traj, rng)
    if result.method == "mhe":
        return None
    return rts(result)


def monte_carlo(problem, methods, n_runs=50, T=None, seed=0, theta=None, smoother=False,
                progress=None, run_ids=None, **opts):
    """Independent truth/measurement draws per run, all methods on the same data.

    Returns {method: dict(nees (R,T), nis (R,T), sq_err (R,T,nx), rmse (R,), time (R,),
    diverged (R,) bool)} plus summary statistics in out['_summary'].

    run_ids selects a subset of the n_runs independent runs (run r always uses the r-th
    spawned seed), so chunks computed in parallel and joined with `merge_mc` reproduce the
    serial result exactly."""
    model = problem.model(theta)
    T = T or problem.T
    root = np.random.SeedSequence(seed)
    all_seeds = root.spawn(n_runs)
    run_ids = list(range(n_runs)) if run_ids is None else list(run_ids)
    seeds = [all_seeds[i] for i in run_ids]
    n_runs = len(run_ids)
    out = {m: {"nees": np.full((n_runs, T), np.nan), "nis": np.full((n_runs, T), np.nan),
               "sq_err": np.full((n_runs, T, model.nx), np.nan), "rmse": np.full(n_runs, np.nan),
               "time": np.zeros(n_runs), "diverged": np.zeros(n_runs, bool)}
           for m in methods if applicable(m, model)}
    if smoother:
        for m in list(out):
            if m != "mhe":
                out[m + "+s"] = {k: v.copy() for k, v in out[m].items()}
    for r in range(n_runs):
        rg = np.random.default_rng(seeds[r])
        xs, ys = model.simulate(T, rg)
        for m in [k for k in out if not k.endswith("+s")]:
            t0 = time.perf_counter()
            try:
                res = run_method(m, model, ys, rg, **opts)
                ok = np.all(np.isfinite(res.m))
            except Exception:
                res, ok = None, False
            out[m]["time"][r] = time.perf_counter() - t0
            targets = [(m, res)]
            if smoother and m + "+s" in out:
                sm = None
                if ok:
                    try:
                        t1 = time.perf_counter()
                        sm = smooth(res, model, rg)
                        out[m + "+s"]["time"][r] = out[m]["time"][r] + time.perf_counter() - t1
                    except Exception:
                        sm = None
                targets.append((m + "+s", sm))
            for key, rr in targets:
                if rr is None or not np.all(np.isfinite(rr.m)):
                    out[key]["diverged"][r] = True
                    continue
                e = xs[1:] - rr.m[1:]
                out[key]["sq_err"][r] = e * e
                out[key]["rmse"][r] = np.sqrt(np.mean(np.sum(e * e, 1)))
                out[key]["nees"][r] = cs.nees(xs, rr.m, rr.P)
                v, S = getattr(rr, "v", None), getattr(rr, "S", None)
                if v is not None and S is not None and np.all(np.isfinite(S)):
                    out[key]["nis"][r] = cs.nis(v, S)
        if progress is not None:
            progress(r + 1, n_runs)
    out["_summary"] = summarize_mc(out, model.nx, model.ny)
    out["_meta"] = {"problem": problem.name, "n_runs": n_runs, "T": T, "nx": model.nx,
                    "ny": model.ny, "seed": seed, "run_ids": run_ids}
    return out


def merge_mc(parts):
    """Join Monte Carlo chunks (from monte_carlo(..., run_ids=...)) in run order."""
    parts = sorted(parts, key=lambda p: p["_meta"]["run_ids"][0])
    meta = dict(parts[0]["_meta"])
    out = {}
    for m in [k for k in parts[0] if not k.startswith("_")]:
        out[m] = {k: np.concatenate([p[m][k] for p in parts]) for k in parts[0][m]}
    meta["run_ids"] = [i for p in parts for i in p["_meta"]["run_ids"]]
    meta["n_runs"] = len(meta["run_ids"])
    out["_summary"] = summarize_mc(out, meta["nx"], meta["ny"])
    out["_meta"] = meta
    return out


def summarize_mc(out, nx, ny, alpha=0.05):
    summ = {}
    for m, d in out.items():
        if m.startswith("_"):
            continue
        ok = ~d["diverged"]
        R = int(ok.sum())
        s = {"n_ok": R, "n_diverged": int((~ok).sum()), "time_ms": 1e3 * float(np.mean(d["time"]))}
        if R:
            s["rmse"] = float(np.sqrt(np.mean(np.nansum(d["sq_err"][ok], 2))))
            s["rmse_per_state"] = np.sqrt(np.nanmean(d["sq_err"][ok], axis=(0, 1))).tolist()
            anees_t = np.nanmean(d["nees"][ok], 0)
            lo, hi = chi2.ppf([alpha / 2, 1 - alpha / 2], R * nx) / R
            s["anees"] = float(np.nanmean(anees_t))
            s["anees_median"] = float(np.nanmedian(anees_t))
            s["anees_band"] = (float(lo), float(hi))
            s["anees_frac_in"] = float(np.mean((anees_t >= lo) & (anees_t <= hi)))
            if np.any(np.isfinite(d["nis"][ok])):
                anis_t = np.nanmean(d["nis"][ok], 0)
                lo2, hi2 = chi2.ppf([alpha / 2, 1 - alpha / 2], R * ny) / R
                s["anis"] = float(np.nanmean(anis_t))
                s["anis_frac_in"] = float(np.mean((anis_t >= lo2) & (anis_t <= hi2)))
        summ[m] = s
    return summ
