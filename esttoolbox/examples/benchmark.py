"""Benchmarks: timings, MCMC efficiency and interval coverage on the static problems, plus a
Monte Carlo comparison of state estimators.

    python -m esttoolbox.examples.benchmark --out docs

Timings are wall clock on one core; MCMC efficiency is the minimum over parameters of bulk ESS per second, all ESS
computed with the same rank-normalized multi-chain estimator (core.diagnostics).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

def run_static(names, reps=5, cov_reps=40):
    from esttoolbox.core.paramest import (fit_am, fit_bootstrap, fit_hmc, fit_least_squares,
                                          fit_profile)
    from esttoolbox.core.static import get_static
    res = {}
    for n in names:
        p = get_static(n)
        y, sig = p.simulate(rng=np.random.default_rng(0))
        t = []
        for _ in range(reps):
            t0 = time.perf_counter(); r = fit_least_squares(p, y, sig, n_starts=1)
            t.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        for _ in range(50):
            p.forward(p.theta_true)
        tf = (time.perf_counter() - t0) / 50
        h = fit_hmc(p, y, sig, rng=1)
        a = fit_am(p, y, sig, rng=1)
        t0 = time.perf_counter(); fit_bootstrap(p, y, sig, n_boot=100, rng=2); tb = time.perf_counter() - t0
        cov = None
        if n in ("oscillator", "heat", "optical-blur", "acoustic") and cov_reps > 0:
            hit_ls, hit_h, hit_p = [], [], []
            rng = np.random.default_rng(100)
            for _ in range(cov_reps):
                yy, _ = p.simulate(rng=rng)
                q = fit_least_squares(p, yy, sig, n_starts=1)
                hit_ls.append(np.abs(q.theta_hat - p.theta_true) <= 1.645 * q.stderr)
                hh = fit_hmc(p, yy, sig, n_iter=700, burn=300, rng=rng)
                hit_h.append((hh.ci[:, 0] <= p.theta_true) & (p.theta_true <= hh.ci[:, 1]))
                pr = fit_profile(p, yy, sig, level=0.90)
                hit_p.append((pr.ci[:, 0] <= p.theta_true) & (p.theta_true <= pr.ci[:, 1]))
            cov = {"ls": float(np.mean(hit_ls)), "hmc": float(np.mean(hit_h)),
                   "profile": float(np.mean(hit_p))}
        res[n] = {"ls_time": float(np.median(t)), "forward_time": tf,
                  "hmc_time": h.runtime, "hmc_ess_min": float(np.min(h.extras["ess_bulk"])),
                  "hmc_rhat_max": float(np.max(h.extras["rhat"])),
                  "am_time": a.runtime, "am_ess_min": float(np.min(a.extras["ess_bulk"])),
                  "boot100_time": tb, "coverage90": cov, "ls_theta": r.theta_hat.tolist()}
    return res


def state_mc(n_runs=50, seed=0, only=None):
    from esttoolbox.core.run import monte_carlo
    from esttoolbox.core.systems import get_ssm
    plan = {
        "cv-tracking": ["kf", "srkf", "ukf", "enkf", "pf"],
        "pendulum": ["ekf", "iekf", "ukf", "ckf", "ghkf", "pf"],
        "ct-radar": ["ekf", "iekf", "ukf", "ckf", "iplf", "etkf", "pf", "rpf", "pf-ekf"],
        "ungm": ["ekf", "ukf", "ckf", "enkf", "pf", "pf-ekf"],
        "lorenz63": ["ekf", "ukf", "enkf", "etkf", "pf"],
        "stochastic-volatility": ["pf", "rpf"],
        "reactor": ["ekf", "ukf", "ckf", "mhe", "pf", "rpf"],
    }
    out = {}
    for name, methods in plan.items():
        if only and name not in only:
            continue
        t0 = time.perf_counter()
        runs = n_runs if name != "reactor" else max(10, n_runs // 3)
        mc = monte_carlo(get_ssm(name), methods, runs, seed=seed, smoother=True, horizon=15)
        out[name] = {"meta": mc["_meta"], "summary": mc["_summary"],
                     "nees": {m: np.nanmean(d["nees"][~d["diverged"]], 0).tolist()
                              for m, d in mc.items() if not m.startswith("_")
                              and (~d["diverged"]).any()}}
        print(f"  {name}: {time.perf_counter() - t0:.1f} s")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--coverage-reps", type=int, default=40)
    ap.add_argument("--skip-state", action="store_true")
    ap.add_argument("--skip-static", action="store_true")
    ap.add_argument("--state-systems", default=None, help="comma list (default: all)")
    a = ap.parse_args(argv)
    names = ["oscillator", "heat", "projectile", "reaction-diffusion", "optical-blur", "acoustic"]
    out = Path(a.out); (out / "benchmarks").mkdir(parents=True, exist_ok=True)
    path = out / "benchmarks" / "benchmarks.json"
    report = json.load(open(path)) if path.exists() else {}   # update in place
    report.update({"python": sys.version.split()[0], "numpy": np.__version__})
    if not a.skip_static:
        print("static benchmarks …")
        report["static"] = run_static(names, cov_reps=a.coverage_reps)
    if not a.skip_state:
        print("state-estimation Monte Carlo …")
        only = a.state_systems.split(",") if a.state_systems else None
        report.setdefault("state", {}).update(state_mc(a.runs, only=only))
    with open(out / "benchmarks" / "benchmarks.json", "w") as fh:
        json.dump(report, fh, indent=1, default=lambda o: None)
    print_tables(report)
    return report


def print_tables(rep):
    st = rep.get("static", {})
    if st:
        print("\nstatic problems")
        print(f"{'problem':20s} {'LS ms':>7s} {'fwd µs':>7s} {'HMC ESS/s':>10s} {'AM ESS/s':>9s} "
              f"{'boot100 s':>9s}")
    for n, b in st.items():
        print(f"{n:20s} {1e3 * b['ls_time']:7.1f} {1e6 * b['forward_time']:7.0f} "
              f"{b['hmc_ess_min'] / b['hmc_time']:10.0f} {b['am_ess_min'] / b['am_time']:9.0f} "
              f"{b['boot100_time']:9.2f}")
    if "state" in rep:
        for name, d in rep["state"].items():
            print(f"\n{name} ({d['meta']['n_runs']} runs, T = {d['meta']['T']})")
            for m, s in d["summary"].items():
                if not s.get("n_ok"):
                    print(f"  {m:12s} all diverged"); continue
                print(f"  {m:12s} RMSE {s['rmse']:9.4g}  ANEES med {s['anees_median']:8.3g}  "
                      f"in-band {100 * s['anees_frac_in']:4.0f}%  div {s['n_diverged']:3d}  "
                      f"{s['time_ms']:8.1f} ms")


if __name__ == "__main__":
    main()
