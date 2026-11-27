"""esttoolbox command line.

  esttoolbox list
  esttoolbox filter  --system ungm --methods ekf,ukf,pf,pf-ekf --smooth --seed 1
  esttoolbox mc      --system ct-radar --methods ekf,ukf,ckf,pf --runs 50 --smooth
  esttoolbox sysid   --system cv-tracking --method em
  esttoolbox sysid   --system stochastic-volatility --method pmmh --n-iter 3000
  esttoolbox static  --system oscillator --method all

Your own data (see core/userdata.py for the file layout):
  esttoolbox filter  --data track.csv --system cv --methods kf,ukf --smooth
  esttoolbox sysid   --data track.csv --system cv --method em
  esttoolbox static  --data decay.csv --system exp-decay --method least-squares,hmc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if not str(k).startswith("_arr")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 64 else f"<array {o.shape}>"
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _opts(a):
    return dict(n_particles=a.n_particles, n_ens=a.n_ens, horizon=a.horizon,
                inflation=a.inflation, alpha=a.ukf_alpha, resampling=a.resampling)


def _slug(name):
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")


def _save(fig, path):
    fig.savefig(path, dpi=130, bbox_inches="tight")


def cmd_list(a):
    from .core.systems import list_ssm
    from .core.static import list_static
    from .core.run import STATE_METHODS
    from .core.paramest import METHODS
    print("state-space systems (filter / mc / sysid):")
    for n, p in list_ssm().items():
        print(f"  {n:22s} {p.description}\n  {'':22s} params: {', '.join(p.param_names)}")
    print("\nstate estimators:")
    for k, v in STATE_METHODS.items():
        print(f"  {k:10s} {v}")
    print("  (+ --smooth: RTS-type smoother for Gaussian/ensemble filters, FFBSi for PFs;"
          " batch-map: full-horizon MAP)")
    print("\nstatic problems (static):")
    for n, p in list_static().items():
        print(f"  {n:22s} {p.description}")
    print("\nstatic methods: " + ", ".join(METHODS))
    print("sysid methods: pem, em (linear models), pmmh")
    return 0


def _series_problem(a):
    """(problem, model, xs, ys, T, label) for --data or a simulated run."""
    from .core.systems import get_ssm
    from .core.userdata import KINEMATIC, full_truth, kinematic_problem, load_series
    if a.data:
        S = load_series(a.data)
        for w in S.warnings:
            print(f"note: {w}")
        name = a.system or "cv"
        p = kinematic_problem(S, name) if name in KINEMATIC else get_ssm(name)
        model = p.model()
        if model.ny != S.ny:
            sys.exit(f"{p.name} expects {model.ny} measurement channels, {a.data} has {S.ny}")
        xs = full_truth(S, model)
        if xs is None:
            xs = np.full((S.T + 1, model.nx), np.nan)
        return p, model, xs, S.ys, S.T, f"{Path(a.data).name} ({S.describe()})"
    if not a.system:
        sys.exit("--system is required without --data")
    p = get_ssm(a.system)
    model = p.model()
    T = a.T or p.T
    xs, ys = model.simulate(T, np.random.default_rng(a.seed))
    return p, model, xs, ys, T, f"simulated, T = {T}, seed = {a.seed}"


def _filter_table(xs, results):
    from .core import consistency as cs
    hdr = (f"{'method':14s} {'RMSE':>9s} {'ANEES':>8s} {'band95':>15s} {'%in':>5s} {'ANIS':>7s}"
           f" {'LB p':>6s} {'loglik':>10s} {'time ms':>8s}")
    print(hdr); print("-" * len(hdr))
    rows = {}
    for n, r in results.items():
        rep = cs.report(xs, r)
        rows[n] = rep
        if rep.get("diverged"):
            print(f"{n:14s} {'diverged':>9s}")
            continue
        if "anees" not in rep:                 # no ground truth
            lb = np.min(rep["ljung_box_p"]) if "ljung_box_p" in rep else np.nan
            ll = getattr(r, "loglik", np.nan)
            print(f"{n:14s} {'—':>9s} {'—':>8s} {'':>15s} {'':>5s} {rep.get('anis', np.nan):7.3g}"
                  f" {lb:6.3f} {ll if ll is not None else np.nan:10.2f} {1e3 * r.runtime:8.1f}")
            continue
        lo, hi = rep["anees_band"]
        lb = np.min(rep["ljung_box_p"]) if "ljung_box_p" in rep else np.nan
        ll = getattr(r, "loglik", np.nan)
        print(f"{n:14s} {rep['rmse']:9.4g} {rep['anees']:8.3g} [{lo:6.3g},{hi:6.3g}] "
              f"{100 * rep['nees_frac_in']:5.0f} {rep.get('anis', np.nan):7.3g} {lb:6.3f} "
              f"{ll if ll is not None else np.nan:10.2f} {1e3 * r.runtime:8.1f}")
    return rows


def cmd_filter(a):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from .core import plots
    from .core import consistency as cs
    from .core.run import applicable, run_method, smooth
    from .core.mhe import batch_map
    p, model, xs, ys, T, label = _series_problem(a)
    rng = np.random.default_rng(a.seed + 1)
    methods = list(p.recommended) if a.methods == "recommended" else a.methods.split(",")
    print(f"system: {p.name} — {p.description}\ndata: {label}\nnx = {model.nx}, "
          f"ny = {model.ny}\n")
    results = {}
    for m in methods:
        if m == "batch-map":
            results[m] = batch_map(model, ys)
            continue
        if not applicable(m, model):
            print(f"  skipping {m}: not applicable to {p.name}")
            continue
        results[m] = run_method(m, model, ys, rng, **_opts(a))
        if a.smooth:
            s = smooth(results[m], model, rng)
            if s is not None:
                results[s.method] = s
    rows = _filter_table(xs, results)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if not a.no_plots:
        f = Figure(figsize=(10, 2.2 * min(model.nx, 5) + 1)); plots.plot_states(f, model, xs, results)
        _save(f, out / f"{_slug(p.name)}_states.png")
        cols = cs.truth_columns(xs)
        if cols:
            f = Figure(figsize=(10, 6))
            nees = {n: cs.nees(xs[:, cols], r.m[:, cols], r.P[:, cols][:, :, cols])
                    for n, r in results.items() if np.all(np.isfinite(r.m))}
            nis = {n: cs.nis(r.v, r.S) for n, r in results.items()
                   if getattr(r, "v", None) is not None and np.all(np.isfinite(r.S))}
            plots.plot_consistency(f, nees, nis, len(cols), model.ny)
            _save(f, out / f"{_slug(p.name)}_consistency.png")
        if p.xy:
            f = Figure(figsize=(7, 6)); plots.plot_track(f, model, xs, results, p.xy, ys, p)
            _save(f, out / f"{_slug(p.name)}_track.png")
        for n, r in results.items():
            if getattr(r, "particles", None) is not None:
                f = Figure(figsize=(10, 6)); plots.plot_particles(f, model, xs, r)
                _save(f, out / f"{_slug(p.name)}_{n}_particles.png")
                break
    with open(out / "results.json", "w") as fh:
        json.dump(_jsonable({"system": p.name, "data": a.data, "T": T, "seed": a.seed,
                             "results": rows}), fh, indent=2)
    print(f"\nwrote {out}/")
    return 0


def cmd_mc(a):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from .core import plots
    from .core.run import monte_carlo
    from .core.systems import get_ssm
    p = get_ssm(a.system)
    methods = list(p.recommended) if a.methods == "recommended" else a.methods.split(",")
    mc = monte_carlo(p, methods, a.runs, a.T, a.seed, smoother=a.smooth,
                     progress=lambda i, n: print(f"\r  run {i}/{n}", end="", flush=True),
                     **_opts(a))
    print()
    S = mc["_summary"]
    hdr = (f"{'method':12s} {'RMSE':>9s} {'ANEES':>9s} {'med':>7s} {'band95':>15s} {'%in':>5s}"
           f" {'ANIS':>7s} {'div':>4s} {'ms/run':>8s}")
    print(hdr); print("-" * len(hdr))
    for m, s in S.items():
        if not s.get("n_ok"):
            print(f"{m:12s} all runs diverged"); continue
        lo, hi = s["anees_band"]
        print(f"{m:12s} {s['rmse']:9.4g} {s['anees']:9.3g} {s['anees_median']:7.3g} "
              f"[{lo:6.3g},{hi:6.3g}] {100 * s['anees_frac_in']:5.0f} {s.get('anis', np.nan):7.3g}"
              f" {s['n_diverged']:4d} {s['time_ms']:8.1f}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if not a.no_plots:
        md = p.model()
        f = Figure(figsize=(10, 6))
        plots.plot_consistency(f, {m: d["nees"][~d["diverged"]] for m, d in mc.items()
                                   if not m.startswith("_") and (~d["diverged"]).any()},
                               {m: d["nis"][~d["diverged"]] for m, d in mc.items()
                                if not m.startswith("_") and (~d["diverged"]).any()},
                               md.nx, md.ny, n_runs=a.runs)
        _save(f, out / f"{_slug(p.name)}_mc_consistency.png")
        f = Figure(figsize=(12, 4)); plots.plot_mc_summary(f, S, f"{p.name}: {a.runs} runs")
        _save(f, out / f"{_slug(p.name)}_mc_summary.png")
    with open(out / "mc_summary.json", "w") as fh:
        json.dump(_jsonable({"meta": mc["_meta"], "summary": S}), fh, indent=2)
    print(f"\nwrote {out}/")
    return 0


def cmd_sysid(a):
    from .core import sysid
    p, model, xs, ys, T, label = _series_problem(a)
    rng = np.random.default_rng(a.seed + 1)
    known = not a.data
    free = None if a.free is None else [p.param_names.index(n) for n in a.free.split(",")]
    if a.method == "pem":
        r = sysid.pem(p, ys, a.filter, use_prior=a.map, free=free)
    elif a.method == "em":
        if not (p.model().linear and list(p.param_names) == ["q", "r"]):
            sys.exit("em needs a linear-Gaussian model with θ = (q, r): cv-tracking, rw, cv or ca")
        r = sysid.em(p, ys, a.n_iter or 200)
    elif a.method == "pmmh":
        r = sysid.pmmh(p, ys, a.n_particles, a.n_iter or 3000, a.burn, rng=rng, free=free,
                       progress=lambda i, n: print(f"\r  iteration {i}/{n}", end="", flush=True))
        print()
    else:
        sys.exit(f"unknown sysid method {a.method}")
    print(f"system: {p.name}, data: {label}, method: {r.method}, runtime {r.runtime:.2f} s,"
          f" log-likelihood {r.loglik:.3f}")
    hdr = f"{'param':>12s} {'true':>11s} {'estimate':>11s} {'stderr':>10s} {'z':>7s}"
    print(hdr); print("-" * len(hdr))
    for j, n in enumerate(p.param_names):
        s = r.stderr[j]
        t = p.theta_true[j] if known else np.nan
        z = (r.theta_hat[j] - t) / s if s > 0 else np.nan
        print(f"{n:>12s} {t:11.4g} {r.theta_hat[j]:11.4g} {s:10.3g} {z:7.2f}")
    for k in ("accept_rate", "iterations", "nfev", "aic", "bic"):
        if k in r.extras:
            print(f"  {k}: {r.extras[k]:.4g}")
    if "ess_bulk" in r.extras:
        print("  ESS (bulk): " + ", ".join(f"{e:.0f}" for e in r.extras["ess_bulk"]))
    return 0


def cmd_static(a):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from .core import plots
    from .core.paramest import METHODS
    from .core.static import get_static
    prob = get_static(a.system)
    methods = list(METHODS) if a.method == "all" else a.method.split(",")
    rng = np.random.default_rng(a.seed)
    if a.data:
        from dataclasses import replace
        from .core.userdata import load_static
        if not prob.accepts_data:
            sys.exit(f"{prob.name} has a fixed design; with --data use one of: " + ", ".join(
                n for n, q in __import__("esttoolbox.core.static", fromlist=["x"]).list_static()
                .items() if q.accepts_data))
        D = load_static(a.data, prob.inputs)
        prob, y, sigma = prob.with_data(D)
        if D.sigma is None and a.noise is not None:
            sigma = a.noise; prob = replace(prob, noise_std=sigma)
        print(f"problem: {prob.name} — {prob.description}\ndata: {Path(a.data).name} "
              f"({D.describe()}), σ = {sigma:g}" + (" (whitened by the file's σ)"
                                                    if D.sigma is not None else
                                                    " (estimated; set --noise if known)") + "\n")
    else:
        y, sigma = prob.simulate(a.noise, rng)
        print(f"problem: {prob.name} — {prob.description}\nn = {y.size}, σ = {sigma:g}\n")
    results = {}
    for m in methods:
        r = METHODS[m](prob, y, sigma, rng=rng, n_boot=a.n_boot)
        results[m] = r
        print(f"[{m}]  {r.runtime:.3f} s")
        for j, n in enumerate(prob.param_names):
            e, s = r.theta_hat[j], r.stderr[j]
            ci = f"  [{r.ci[j, 0]:.4g}, {r.ci[j, 1]:.4g}]" if r.ci is not None else ""
            if prob.has_truth:
                t = prob.theta_true[j]
                print(f"  {n:>10s} true {t:10.4g}  est {e:10.4g} ± {s:9.3g}  z {(e - t) / s:6.2f}{ci}")
            else:
                print(f"  {n:>10s} est {e:10.4g} ± {s:9.3g}{ci}")
        if "rhat" in r.extras:
            print(f"  R̂ max {np.max(r.extras['rhat']):.3f}, ESS bulk min "
                  f"{np.min(r.extras['ess_bulk']):.0f}, ESS/s min {np.min(r.extras['ess_per_s']):.0f},"
                  f" acceptance {r.extras['accept_rate']:.2f}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if not a.no_plots:
        f = Figure(figsize=(10, 7)); plots.plot_static_fit(f, prob, y, results)
        _save(f, out / f"{prob.name}_fit.png")
        f = Figure(figsize=(10, 2 + 1.6 * prob.dim)); plots.plot_compare(
            f, prob.param_names, prob.theta_true if prob.has_truth else None, results)
        _save(f, out / f"{prob.name}_compare.png")
        for m in ("hmc", "ensemble", "am", "rwm", "bootstrap"):
            if m in results:
                f = Figure(figsize=(8, 8))
                plots.plot_corner(f, prob.param_names, results[m].samples,
                                  prob.theta_true if prob.has_truth else None,
                                  results[m].theta_hat)
                _save(f, out / f"{prob.name}_{m}_corner.png")
                break
    rep = {m: {"theta_hat": r.theta_hat, "stderr": r.stderr, "runtime": r.runtime,
               **{k: v for k, v in r.extras.items() if np.isscalar(v)}}
           for m, r in results.items()}
    with open(out / "results.json", "w") as fh:
        json.dump(_jsonable({"problem": prob.name, "sigma": sigma, "theta_true": prob.theta_true if prob.has_truth else None, "data": a.data,
                             "results": rep}), fh, indent=2)
    print(f"\nwrote {out}/")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="esttoolbox", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list", help="systems and methods")

    def common(sp):
        sp.add_argument("--system", default=None,
                        help="system name; with --data also rw / cv / ca (default cv)")
        sp.add_argument("--data", default=None, help="your measurement file (.csv/.txt/.npy/.npz)")
        sp.add_argument("--T", type=int, default=None, help="horizon (default per system)")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--out", default="est_output")
        sp.add_argument("--no-plots", action="store_true")
        sp.add_argument("--n-particles", type=int, default=1000)
        sp.add_argument("--n-ens", type=int, default=100)
        sp.add_argument("--horizon", type=int, default=10, help="MHE window")
        sp.add_argument("--inflation", type=float, default=1.0)
        sp.add_argument("--ukf-alpha", type=float, default=1.0)
        sp.add_argument("--resampling", default="systematic",
                        choices=["systematic", "stratified", "residual", "multinomial"])

    sp = sub.add_parser("filter", help="one run of several estimators")
    common(sp)
    sp.add_argument("--methods", default="recommended")
    sp.add_argument("--smooth", action="store_true")
    sp = sub.add_parser("mc", help="Monte Carlo consistency / accuracy study")
    common(sp)
    sp.add_argument("--methods", default="recommended")
    sp.add_argument("--runs", type=int, default=50)
    sp.add_argument("--smooth", action="store_true")
    sp = sub.add_parser("sysid", help="state-space parameter estimation")
    common(sp)
    sp.add_argument("--method", default="pem", choices=["pem", "em", "pmmh"])
    sp.add_argument("--filter", default=None, help="pem: kf/ekf/ukf/ckf (default kf or ukf)")
    sp.add_argument("--map", action="store_true", help="pem: include the prior")
    sp.add_argument("--free", default=None, help="comma list of parameters to estimate")
    sp.add_argument("--n-iter", type=int, default=None)
    sp.add_argument("--burn", type=int, default=1000)
    sp = sub.add_parser("static", help="static parameter estimation")
    sp.add_argument("--system", required=True)
    sp.add_argument("--data", default=None, help="your data: input column(s), y, optional sigma")
    sp.add_argument("--method", default="all")
    sp.add_argument("--noise", type=float, default=None)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--n-boot", type=int, default=300)
    sp.add_argument("--out", default="est_output")
    sp.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd is None:
        ap.print_help()
        return 0
    return {"list": cmd_list, "filter": cmd_filter, "mc": cmd_mc, "sysid": cmd_sysid,
            "static": cmd_static}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
