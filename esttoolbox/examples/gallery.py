"""Regenerate the README figures.

    python -m esttoolbox.examples.gallery docs/figures [docs/benchmarks/benchmarks.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from esttoolbox.core import consistency as cs  # noqa: E402
from esttoolbox.core import plots, sysid  # noqa: E402
from esttoolbox.core.mhe import batch_map, run_mhe  # noqa: E402
from esttoolbox.core.paramest import fit_hmc  # noqa: E402
from esttoolbox.core.run import monte_carlo, run_method, smooth  # noqa: E402
from esttoolbox.core.static import get_static  # noqa: E402
from esttoolbox.core.systems import get_ssm  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})


def ct_radar(out):
    p = get_ssm("ct-radar"); md = p.model()
    xs, ys = md.simulate(p.T, 3)
    R = {m: run_method(m, md, ys, 1) for m in ("ekf", "ukf", "pf")}
    R["ukf+rts"] = smooth(R["ukf"], md)
    fig = plt.figure(figsize=(12, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(ys[:, 0] * np.cos(ys[:, 1]), ys[:, 0] * np.sin(ys[:, 1]), ".", color="0.65", ms=3,
            label="range/bearing → xy")
    ax.plot(xs[:, 0], xs[:, 2], "k-", lw=2, label="truth")
    for n, r in R.items():
        ax.plot(r.m[:, 0], r.m[:, 2], "--" if "+" in n else "-", color=plots.color(n), lw=1.2,
                label=n)
    ax.plot(0, 0, "k^", ms=8)
    ax.set_aspect("equal", adjustable="datalim"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("coordinated turn, unknown turn rate, radar at origin")
    a1 = fig.add_subplot(gs[0, 1]); a2 = fig.add_subplot(gs[1, 1])
    t = np.arange(len(xs))
    a1.plot(t, np.rad2deg(xs[:, 4]), "k-", lw=1.6, label="truth")
    for n in ("ekf", "ukf+rts", "pf"):
        r = R[n]; sd = np.sqrt(r.P[:, 4, 4])
        a1.plot(t, np.rad2deg(r.m[:, 4]), color=plots.color(n), lw=1, label=n)
        a1.fill_between(t, np.rad2deg(r.m[:, 4] - 2 * sd), np.rad2deg(r.m[:, 4] + 2 * sd),
                        color=plots.color(n), alpha=0.12)
    a1.set_ylabel("turn rate ω [°/s]"); a1.legend(fontsize=7, ncol=4); a1.grid(alpha=0.3)
    for n, r in R.items():
        a2.semilogy(t[1:], cs.nees(xs, r.m, r.P), color=plots.color(n), lw=0.8,
                    ls="--" if "+" in n else "-", label=n)
    from scipy.stats import chi2
    a2.axhspan(*chi2.ppf([0.025, 0.975], 5), color="g", alpha=0.12)
    a2.set_ylabel("NEES (χ²₅ band)"); a2.set_xlabel("time [s]"); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "ct_radar.png", dpi=130); plt.close(fig)


def ungm(out):
    p = get_ssm("ungm"); md = p.model()
    xs, ys = md.simulate(p.T, 7)
    pf = run_method("pf", md, ys, 1, n_particles=2000)
    fb = smooth(pf, md, 2, n_traj=200)
    ekf, ukf = run_method("ekf", md, ys), run_method("ukf", md, ys)
    fig = plt.figure(figsize=(12, 5))
    plots.plot_particles(fig, md, xs, pf, 0, fb)
    ax = fig.axes[0]
    ax.plot(ekf.m[:, 0], color=plots.color("ekf"), lw=1, label="ekf")
    ax.plot(ukf.m[:, 0], color=plots.color("ukf"), lw=1, label="ukf")
    ax.set_ylim(-35, 35)
    ax.legend(fontsize=8, ncol=5, loc="upper right")
    for n, r in (("ekf", ekf), ("ukf", ukf), ("pf", pf), ("pf+ffbsi", fb)):
        print(f"    ungm {n}: RMSE {cs.rmse(xs, r.m):.3f}")
    fig.suptitle("UNGM: y = x²/20 hides the sign of x — the filtering density is bimodal")
    fig.tight_layout(); fig.savefig(out / "ungm_particles.png", dpi=130); plt.close(fig)


def reactor(out):
    p = get_ssm("reactor"); md = p.model()
    xs, ys = md.simulate(p.T, 0)
    R = {"ekf": run_method("ekf", md, ys), "ckf": run_method("ckf", md, ys),
         "mhe (x ≥ 0, N = 15)": run_mhe(md, ys, 15),
         "mhe (unconstrained)": run_mhe(md, ys, 15, constrained=False),
         "batch MAP (x ≥ 0)": batch_map(md, ys)}
    cols = {"ekf": plots.color("ekf"), "ckf": plots.color("ckf"), "mhe (x ≥ 0, N = 15)": "#637939",
            "mhe (unconstrained)": "#b5cf6b", "batch MAP (x ≥ 0)": "k"}
    t = np.arange(len(xs)) * md.dt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, ax in enumerate(axes):
        ax.plot(t, xs[:, i], color="0.3", lw=3, alpha=0.4, label="truth")
        for n, r in R.items():
            ax.plot(t, r.m[:, i], color=cols[n], lw=1.2, ls=":" if "MAP" in n else "-", label=n)
        ax.axhline(0, color="r", lw=0.6)
        ax.set_xlabel("time"); ax.set_ylabel(md.state_names[i]); ax.grid(alpha=0.3)
    axes[0].set_ylim(-4, 6); axes[1].set_ylim(-2, 7)
    axes[1].legend(fontsize=8)
    fig.suptitle("2A → B reactor, total-pressure measurement, prior (0.1, 4.5) vs truth (3, 1): "
                 "EKF converges to negative pressures; constrained MHE does not")
    fig.tight_layout(); fig.savefig(out / "reactor_mhe.png", dpi=130); plt.close(fig)


def consistency(out):
    p = get_ssm("ct-radar")
    mc = monte_carlo(p, ["ekf", "ukf", "ckf", "pf", "pf-ekf"], 50, seed=5, smoother=True)
    md = p.model()
    fig = plt.figure(figsize=(11, 6.5))
    sel = ("ekf", "ukf", "pf", "pf-ekf", "ukf+s")
    plots.plot_consistency(fig, {m: mc[m]["nees"][~mc[m]["diverged"]] for m in sel},
                           {m: mc[m]["nis"][~mc[m]["diverged"]] for m in sel}, md.nx, md.ny, 50)
    fig.savefig(out / "consistency_mc.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    fig = plt.figure(figsize=(12, 4))
    plots.plot_mc_summary(fig, mc["_summary"], "ct-radar: 50 Monte Carlo runs (+s = smoothed)")
    fig.tight_layout(); fig.savefig(out / "mc_summary.png", dpi=130); plt.close(fig)


def lorenz(out):
    p = get_ssm("lorenz63"); md = p.model()
    xs, ys = md.simulate(p.T, 2)
    R = {m: run_method(m, md, ys, 3, n_ens=50) for m in ("ekf", "ukf", "etkf")}
    R["etkf+rts"] = smooth(R["etkf"], md)
    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.6])
    ax = fig.add_subplot(gs[:, 0], projection="3d")
    ax.plot(*xs.T, "k-", lw=0.8, label="truth")
    ax.plot(*R["etkf+rts"].m.T, color=plots.color("etkf"), lw=0.8, ls="--", label="etkf+rts")
    ax.set_title("Lorenz-63 attractor"); ax.legend(fontsize=8)
    t = np.arange(len(xs)) * md.dt
    for i in range(3):
        a = fig.add_subplot(gs[i, 1])
        a.plot(t[1:], ys[:, i], ".", color="0.7", ms=2)
        a.plot(t, xs[:, i], "k-", lw=1.4)
        for n, r in R.items():
            a.plot(t, r.m[:, i], color=plots.color(n), lw=0.9, ls="--" if "+" in n else "-",
                   label=f"{n} ({cs.rmse(xs, r.m):.2f})" if i == 0 else None)
        a.set_ylabel("xyz"[i]); a.grid(alpha=0.3)
    fig.axes[1].legend(fontsize=7, ncol=4, title="method (RMSE)", title_fontsize=7)
    fig.tight_layout(); fig.savefig(out / "lorenz63.png", dpi=130); plt.close(fig)


def sysid_fig(out):
    p = get_ssm("cv-tracking")
    xs, ys = p.model().simulate(300, 4)
    em = sysid.em(p, ys, 300)
    pe = sysid.pem(p, ys, "kf")
    sv = get_ssm("stochastic-volatility")
    _, ysv = sv.model().simulate(500, 1)
    pm = sysid.pmmh(sv, ysv, 300, 6000, 1500, rng=0)
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(3, 4)
    a = fig.add_subplot(gs[0, :2])
    a.plot(np.arange(1, len(em.history) + 1), em.history, "o-", ms=2, label="EM")
    a.axhline(pe.loglik, color="r", ls="--", label=f"PEM optimum ({pe.loglik:.2f})")
    a.set_xscale("log"); a.set_xlabel("EM iteration"); a.set_ylabel("log-likelihood")
    a.set_title(f"cv-tracking: EM (q̂={em.theta_hat[0]:.4f}, r̂={em.theta_hat[1]:.3f}) vs "
                f"PEM (q̂={pe.theta_hat[0]:.4f}±{pe.stderr[0]:.4f}); truth (0.05, 1.0)",
                fontsize=8)
    a.legend(fontsize=8); a.grid(alpha=0.3)
    a = fig.add_subplot(gs[0, 2:])
    a.plot(pm.history, lw=0.5); a.set_title("PMMH: PF log-likelihood estimates (post burn-in)")
    a.grid(alpha=0.3)
    for j, n in enumerate(sv.param_names):
        a = fig.add_subplot(gs[1, j]) if j < 3 else None
        a.plot(pm.samples[:, j], lw=0.4); a.axhline(sv.theta_true[j], color="k", ls="--")
        a.set_title(f"{n}: {pm.theta_hat[j]:.3f} ± {pm.stderr[j]:.3f} (true {sv.theta_true[j]})",
                    fontsize=8)
        b = fig.add_subplot(gs[2, j])
        b.hist(pm.samples[:, j], 40, density=True, color="tab:red", alpha=0.7)
        b.axvline(sv.theta_true[j], color="k", ls="--"); b.set_xlabel(n)
    a = fig.add_subplot(gs[1:, 3])
    a.plot(pm.samples[:, 0], pm.samples[:, 1], ".", ms=1, alpha=0.4)
    a.plot(sv.theta_true[0], sv.theta_true[1], "k+", ms=12, mew=2)
    a.set_xlabel("phi"); a.set_ylabel("sigma"); a.set_title("joint (φ, σ)")
    fig.suptitle(f"stochastic volatility, T = 500: PMMH with 300 particles, "
                 f"acceptance {pm.extras['accept_rate']:.2f}, ESS "
                 + ", ".join(f"{e:.0f}" for e in pm.extras["ess_bulk"]))
    fig.tight_layout(); fig.savefig(out / "sysid.png", dpi=130); plt.close(fig)


def static_posterior(out):
    prob = get_static("acoustic")
    y, s = prob.simulate(rng=0)
    r = fit_hmc(prob, y, s, rng=1)
    fig = plt.figure(figsize=(8, 8))
    plots.plot_corner(fig, prob.param_names, r.samples, prob.theta_true, r.theta_hat, c="tab:red")
    fig.suptitle(f"acoustic source: HMC posterior (4 chains, R̂ ≤ {np.max(r.extras['rhat']):.3f},"
                 f" min bulk ESS {np.min(r.extras['ess_bulk']):.0f} in {r.runtime:.1f} s)")
    fig.savefig(out / "static_posterior.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def benchmark_figs(out, bench):
    rep = json.load(open(bench))
    st = rep["static"]
    names = list(st)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    x = np.arange(len(names)); w = 0.38
    axes[0].bar(x - w / 2, [st[n]["am_ess_min"] / st[n]["am_time"] for n in names], w,
                color=plots.color("am"), label="adaptive Metropolis (4 chains)")
    axes[0].bar(x + w / 2, [st[n]["hmc_ess_min"] / st[n]["hmc_time"] for n in names], w,
                color=plots.color("hmc"), label="HMC (4 chains)")
    axes[0].set_yscale("log"); axes[0].set_title("MCMC efficiency: min bulk ESS / s")
    axes[1].bar(x - w / 2, [1e6 * st[n]["forward_time"] for n in names], w, color="tab:blue",
                label="single θ")
    if "forward_batch16_per_theta" in st[names[0]]:
        axes[1].bar(x + w / 2, [1e6 * st[n]["forward_batch16_per_theta"] for n in names], w,
                    color="tab:cyan", label="per θ in a batch of 16")
    axes[2].bar(x, [1e3 * st[n]["ls_time"] for n in names], 0.6, color="tab:blue")
    axes[1].set_title("forward model evaluation [µs per θ]"); axes[1].set_yscale("log")
    axes[2].set_title("least squares (single start) [ms]"); axes[2].set_yscale("log")
    for a in axes:
        a.set_xticks(x); a.set_xticklabels(names, rotation=30, fontsize=8)
        if a.get_legend_handles_labels()[0]:
            lo, hi = a.get_ylim(); a.set_ylim(lo, hi * 6)      # headroom for the legend
            a.legend(fontsize=8, loc="upper right")
        a.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out / "benchmark_static.png", dpi=130); plt.close(fig)
    cov_names = [n for n in names if st[n]["coverage90"]]
    if cov_names:
        fig, ax = plt.subplots(figsize=(9, 3.6))
        series = [("LS ±1.645σ", lambda n: st[n]["coverage90"]["ls"], "0.6"),
                  ("profile likelihood", lambda n: st[n]["coverage90"]["profile"],
                    plots.color("profile")),
                  ("HMC 5–95 %", lambda n: st[n]["coverage90"]["hmc"], plots.color("hmc"))]
        x = np.arange(len(cov_names)); w = 0.8 / len(series)
        for i, (lab, f, c) in enumerate(series):
            ax.bar(x + (i - (len(series) - 1) / 2) * w, [f(n) for n in cov_names], w, color=c,
                   label=lab)
        from esttoolbox.core.static import get_static
        R = rep.get("coverage_reps", 40)
        for i, n in enumerate(cov_names):     # ±2 binomial s.e. for R·p pooled intervals
            se = np.sqrt(0.9 * 0.1 / (R * get_static(n).dim))
            ax.fill_between([i - 0.45, i + 0.45], 0.9 - 2 * se, 0.9 + 2 * se, color="g",
                            alpha=0.15, lw=0, zorder=0)
        ax.axhline(0.9, color="g", lw=1)
        ax.set_ylim(0.6, 1.0); ax.set_xticks(x); ax.set_xticklabels(cov_names)
        ax.set_ylabel("empirical coverage"); ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
        ax.set_title("coverage of nominal 90 % intervals over 40 data sets (parameters pooled; "
                     "band = ±2 binomial s.e.)", fontsize=9)
        fig.tight_layout(); fig.savefig(out / "coverage.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    if "state" in rep:
        st = rep["state"]
        fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
        for ax, (name, d) in zip(axes.ravel(), st.items()):
            S = {m: s for m, s in d["summary"].items() if s.get("n_ok")}
            ms = list(S)
            ax.bar(range(len(ms)), [S[m]["rmse"] for m in ms], color=[plots.color(m) for m in ms],
                   hatch=None)
            for i, m in enumerate(ms):
                if "+" in m:
                    ax.patches[i].set_hatch("//")
                if S[m]["n_diverged"]:
                    ax.text(i, S[m]["rmse"], f"{S[m]['n_diverged']}✗", ha="center", va="bottom",
                            color="r", fontsize=7)
            ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms, rotation=70, fontsize=7)
            ax.set_yscale("log"); ax.set_title(f"{name} ({d['meta']['n_runs']} runs)", fontsize=9)
            ax.grid(alpha=0.3, axis="y")
        axes[0, 0].set_ylabel("RMSE"); axes[1, 0].set_ylabel("RMSE")
        axes.ravel()[-1].axis("off")
        fig.suptitle("state estimation: Monte Carlo RMSE (hatched = smoothed, ✗ = diverged runs)")
        fig.tight_layout(); fig.savefig(out / "benchmark_state.png", dpi=130); plt.close(fig)


def systems_gallery(out):
    """Six of the state-space systems, one panel each."""
    from scipy.stats import chi2  # noqa: F401
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    # bearings-only: range becomes observable after the ownship turns
    p = get_ssm("bearings-only"); md = p.model(); xs, ys = md.simulate(p.T, 2)
    ax = axes[0, 0]
    own = p.landmarks["ownship"]
    ax.plot(own[:, 0], own[:, 1], ":", color="0.3", label="ownship")
    ax.plot(xs[:, 0], xs[:, 1], "k-", lw=2, label="target")
    for m in ("ekf", "ukf", "rpf"):
        r = run_method(m, md, ys, 1, n_particles=3000)
        ax.plot(r.m[:, 0], r.m[:, 1], color=plots.color(m), lw=1.2, label=m)
    ax.set_aspect("equal", adjustable="datalim"); ax.legend(fontsize=7)
    ax.set_title("bearings-only TMA (ownship turns at k = 30, 60)")
    # robot localization
    p = get_ssm("robot-landmarks"); md = p.model(); xs, ys = md.simulate(p.T, 0)
    ax = axes[0, 1]
    ax.plot(*p.landmarks["landmarks"].T, "k^", ms=8, label="landmarks")
    ax.plot(xs[:, 0], xs[:, 1], "k-", lw=2, label="truth")
    r = run_method("ekf", md, ys)
    ax.plot(r.m[:, 0], r.m[:, 1], color=plots.color("ekf"), lw=1.2, label="ekf")
    for k in range(0, p.T + 1, 25):
        from esttoolbox.gui.widgets import ellipse
        ex, ey = ellipse(r.m[k, :2], 25 * r.P[k, :2, :2])
        ax.plot(ex, ey, color=plots.color("ekf"), lw=0.7)
    ax.set_aspect("equal", adjustable="datalim"); ax.legend(fontsize=7)
    ax.set_title("robot localization (EKF, 10σ ellipses every 25 steps)")
    # Lorenz-96 Hovmöller: truth vs ETKF error
    p = get_ssm("lorenz96"); md = p.model(); xs, ys = md.simulate(p.T, 0)
    r = run_method("etkf", md, ys, 0, n_ens=60, inflation=1.05)
    ax = axes[0, 2]
    im = ax.imshow((xs - r.m).T, aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4,
                   extent=[0, p.T * md.dt, 40.5, 0.5])
    ax.set_xlabel("time"); ax.set_ylabel("variable")
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title(f"Lorenz-96 (40-D, 20 observed, σ = 1): ETKF error, "
                 f"RMSE/var {cs.rmse(xs, r.m) / np.sqrt(40):.2f}")
    # battery SoC from the wrong initial guess
    p = get_ssm("battery-soc"); md = p.model(); xs, ys = md.simulate(p.T, 0)
    ax = axes[1, 0]
    t = np.arange(len(xs)) * md.dt / 60
    ax.plot(t, xs[:, 0], "k-", lw=2, label="truth")
    for m in ("ekf", "ukf", "mhe"):
        r = run_method(m, md, ys, 0)
        sd = np.sqrt(r.P[:, 0, 0])
        ax.plot(t, r.m[:, 0], color=plots.color(m), lw=1.2, label=m)
        ax.fill_between(t, r.m[:, 0] - 2 * sd, r.m[:, 0] + 2 * sd, color=plots.color(m), alpha=0.12)
    ax.set_ylim(0.2, 1.05); ax.set_xlabel("time [min]"); ax.set_ylabel("state of charge")
    ax.legend(fontsize=7); ax.set_title("Li-ion SoC, initial guess 0.6 (truth 0.95)")
    # predator-prey with Poisson counts
    p = get_ssm("predator-prey"); md = p.model(); xs, ys = md.simulate(p.T, 1)
    r = run_method("pf", md, ys, 0, n_particles=3000)
    ax = axes[1, 1]
    t = np.arange(len(xs)) * md.dt
    for i, c in ((0, "tab:green"), (1, "tab:red")):
        ax.plot(t[1:], ys[:, i], ".", color=c, ms=4, alpha=0.6)
        ax.plot(t, np.exp(xs[:, i]), "-", color="k", lw=1.2)
        sd = np.sqrt(r.P[:, i, i])
        ax.plot(t, np.exp(r.m[:, i]), color=c, lw=1.2, label=md.state_names[i].replace("log ", ""))
        ax.fill_between(t, np.exp(r.m[:, i] - 2 * sd), np.exp(r.m[:, i] + 2 * sd), color=c, alpha=0.15)
    ax.set_xlabel("time [years]"); ax.set_ylabel("population"); ax.legend(fontsize=7)
    ax.set_title("stochastic Lotka–Volterra, Poisson counts (PF)")
    # re-entry: drag coefficient learning
    p = get_ssm("reentry"); md = p.model(); xs, ys = md.simulate(p.T, 0)
    ax = axes[1, 2]
    t = np.arange(len(xs)) * md.dt
    ax.plot(t, xs[:, 4], "k-", lw=2, label="truth")
    for m in ("ekf", "ukf", "enkf"):
        r = run_method(m, md, ys, 0, n_ens=200)
        sd = np.sqrt(r.P[:, 4, 4])
        ax.plot(t, r.m[:, 4], color=plots.color(m), lw=1.2, label=m)
        ax.fill_between(t, r.m[:, 4] - 2 * sd, r.m[:, 4] + 2 * sd, color=plots.color(m), alpha=0.12)
    ax.set_ylim(-1.5, 2.5); ax.set_xlabel("time [s]"); ax.set_ylabel("log drag coefficient")
    ax.legend(fontsize=7); ax.set_title("ballistic re-entry: unknown drag (Julier & Uhlmann)")
    for a in axes.ravel():
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "systems.png", dpi=120); plt.close(fig)


def static_gallery(out):
    """Six static problems: data, truth and least-squares fit."""
    from esttoolbox.core.paramest import fit_least_squares
    names = ["enzyme-inhibition", "pharmacokinetics", "spectral-peaks", "radial-velocity",
             "sir-epidemic", "theis-well"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6))
    for ax, n in zip(axes.ravel(), names):
        p = get_static(n); y, s = p.simulate(rng=0)
        r = fit_least_squares(p, y, s, rng=0)
        x = np.asarray(p.x, float)
        f, ft = p.forward(r.theta_hat), p.forward(p.theta_true)
        groups = [(None, np.arange(len(x)))] if p.groups is None else \
            [(g, np.flatnonzero(p.groups == g)) for g in dict.fromkeys(p.groups)]
        for c, (g, idx) in enumerate(groups):
            idx = idx[np.argsort(x[idx])]
            ax.plot(x[idx], y[idx], "o", ms=3, color=f"C{c}", label=g or "data")
            ax.plot(x[idx], f[idx], "-", color=f"C{c}", lw=1.3)
            ax.plot(x[idx], ft[idx], "k--", lw=0.8)
        if p.xlog:
            ax.set_xscale("log")
        z = (r.theta_hat - p.theta_true) / r.stderr
        ax.set_title(f"{n} ({p.dim} parameters, max |z| = {np.max(np.abs(z)):.1f})", fontsize=9)
        ax.set_xlabel(p.xlabel); ax.grid(alpha=0.3)
        if p.groups is not None:
            ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "static_problems.png", dpi=120); plt.close(fig)


def mc_speed(problem="pendulum", methods=("ekf", "ukf", "pf"), R=60):
    """Serial monte_carlo vs. the GUI's process pool (chunks of runs on all workers)."""
    import time
    from esttoolbox.core.run import merge_mc
    from esttoolbox.gui.compute import PoolRunner, mc_task, default_workers
    p = get_ssm(problem)
    t = time.perf_counter(); a = monte_carlo(p, list(methods), R, seed=0); ts = time.perf_counter() - t
    P = PoolRunner.pool(); W = default_workers()
    import multiprocessing as mp  # noqa: F401
    chunks = [c.tolist() for c in np.array_split(np.arange(R), 3 * W)]
    res = [P.p.apply_async(mc_task, ((15, 0), problem, list(methods), R, c, None, 0, None, False, {}))
           for c in chunks[:1]]
    [r.get() for r in res]                              # warm the workers
    t = time.perf_counter()
    res = [P.p.apply_async(mc_task, ((15, 0), problem, list(methods), R, c, None, 0, None, False, {}))
           for c in chunks]
    b = merge_mc([r.get() for r in res]); tp = time.perf_counter() - t
    same = all(np.allclose(a[m]["nees"], b[m]["nees"], equal_nan=True) for m in methods)
    return ts, tp, W, same


def _gui_app():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from esttoolbox.gui.app import MainWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    f = app.font(); f.setPointSize(9); app.setFont(f)
    return app, MainWindow


def _settle(app, page, ms=600_000):
    import time
    t = time.time()
    while time.time() - t < ms / 1000:
        app.processEvents()
        deb = getattr(page, "_debounce", None)
        rs = [page.runner] + ([page.mc_runner] if hasattr(page, "mc_runner") else [])
        if not any(r.running() or r.threads for r in rs) and not (deb and deb.isActive()):
            break
        time.sleep(0.005)
    for _ in range(10):
        app.processEvents(); time.sleep(0.005)


def gui_shots(out):
    app, MainWindow = _gui_app()
    w = MainWindow(); w.resize(1600, 960); w.show()
    s = w.state
    s.cb_sys.setCurrentIndex(s.cb_sys.findData("ct-radar")); _settle(app, s)
    s.cb_smooth.setChecked(True); _settle(app, s)
    s.set_cursor_t(30.0); s.tabs.setCurrentWidget(s.v_states); _settle(app, s)
    w.grab().save(str(out / "gui_states.png"))
    s.cb_smooth.setChecked(False)
    s.cb_sys.setCurrentIndex(s.cb_sys.findData("robot-landmarks")); _settle(app, s)
    s.set_cursor_t(40.0); s.tabs.setCurrentWidget(s.v_track); _settle(app, s)
    w.grab().save(str(out / "gui_track.png"))
    s.cb_sys.setCurrentIndex(s.cb_sys.findData("ungm")); _settle(app, s)
    s.cb_smooth.setChecked(True); _settle(app, s)
    s.set_cursor_t(25.0); s.tabs.setCurrentWidget(s.v_part); _settle(app, s)
    w.grab().save(str(out / "gui_particles.png"))
    s.cb_smooth.setChecked(False)
    s.cb_sys.setCurrentIndex(s.cb_sys.findData("battery-soc")); _settle(app, s)
    s.sp_runs.setValue(30); s.run_mc(); _settle(app, s)
    w.grab().save(str(out / "gui_montecarlo.png"))
    w.nav.setCurrentRow(2); st = w.static
    st.cb.setCurrentIndex(st.cb.findData("radial-velocity")); _settle(app, st)
    st.methods.set_checked(["least-squares", "hmc"]); st.run_all(); _settle(app, st)
    st.tabs.setCurrentWidget(st.v_post); _settle(app, st)
    w.grab().save(str(out / "gui_static.png"))
    w.toggle_theme()
    w.nav.setCurrentRow(1); si = w.sysid
    si.cb.setCurrentIndex(si.cb.findData("pendulum")); app.processEvents()
    si.run_all(); _settle(app, si)
    si.tabs.setCurrentWidget(si.v_sl); _settle(app, si)
    w.grab().save(str(out / "gui_sysid_dark.png"))
    w.close()


def example_data(folder=None):
    """Write the two example files used in the README (a manoeuvring target with partial truth
    and a cooling curve with per-point σ) into `folder` (default: examples/data)."""
    folder = Path(folder) if folder else Path(__file__).parent / "data"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    t = np.arange(1, 241) * 0.5
    turn = np.where((t > 40) & (t < 70), 0.06, np.where((t > 90) & (t < 105), -0.1, 0.0))
    hd = np.cumsum(turn * 0.5) + 0.3
    sp = 8.0 + 0.5 * np.sin(t / 15)
    px = np.cumsum(sp * np.cos(hd) * 0.5); py = np.cumsum(sp * np.sin(hd) * 0.5)
    yx, yy = px + 3.0 * rng.standard_normal(len(t)), py + 3.0 * rng.standard_normal(len(t))
    np.savetxt(folder / "track.csv", np.c_[t, yx, yy, px, py], delimiter=",", fmt="%.4f",
               header="t,px,py,x_px,x_py", comments="")
    tm = np.r_[np.linspace(0, 20, 25), np.linspace(22, 90, 20)]
    sig = np.where(tm < 21, 0.15, 0.4)
    T = 21.0 + 68.0 * np.exp(-0.045 * tm) + sig * rng.standard_normal(len(tm))
    np.savetxt(folder / "cooling.csv", np.c_[tm, T, sig], delimiter=",", fmt="%.4f",
               header="t,y,sigma", comments="")
    return folder


def gui_data_shots(out):
    """The GUI on your own data: the example track and cooling curve."""
    app, MainWindow = _gui_app()
    folder = example_data()
    w = MainWindow(); w.resize(1600, 960); w.show()
    s = w.state
    s.source.load(str(folder / "track.csv")); _settle(app, s)
    s.cb_sys.setCurrentIndex(s.cb_sys.findData("constant velocity (2-D)")); _settle(app, s)
    s.methods.set_checked(["kf", "ukf", "pf"]); s.cb_smooth.setChecked(True); _settle(app, s)
    s.set_cursor_t(80.0); s.tabs.setCurrentWidget(s.v_track); _settle(app, s)
    w.grab().save(str(out / "gui_mydata_track.png"))
    w.nav.setCurrentRow(2); st = w.static
    st.source.load(str(folder / "cooling.csv"))
    st.cb.setCurrentIndex(st.cb.findData("exp-decay")); _settle(app, st)
    st.methods.set_checked(["least-squares", "profile", "hmc"]); st.run_all(); _settle(app, st)
    st.tabs.setCurrentWidget(st.v_fit); _settle(app, st)
    w.grab().save(str(out / "gui_mydata_static.png"))
    w.close()


def gui_latency(systems=("cv-tracking", "pendulum", "ct-radar", "robot-landmarks", "battery-soc",
                         "lorenz96"), reps=5):
    """Milliseconds from a parameter change to (a) the first estimate on screen and (b) all
    live estimators on screen, with the default method selection, in the offscreen GUI."""
    import time
    app, MainWindow = _gui_app()
    w = MainWindow(); w.show()
    s = w.state
    rows = []
    for name in systems:
        s.cb_sys.setCurrentIndex(s.cb_sys.findData(name)); _settle(app, s)
        firsts, alls = [], []
        for r in range(reps):
            th = s.params.values() * (1.02 if r % 2 == 0 else 1 / 1.02)
            t0 = time.perf_counter()
            s.params.set_values(th); s._launch()
            first = None
            while True:
                app.processEvents()
                if first is None and s.results:
                    first = time.perf_counter() - t0
                if not s.runner.running() and not s.runner.threads:
                    break
                time.sleep(0.001)
            app.processEvents(); s._do_refresh()
            firsts.append(first if first is not None else np.nan)
            alls.append(time.perf_counter() - t0)
        rows.append((name, ", ".join(s.methods.checked()), 1e3 * np.median(firsts),
                     1e3 * np.median(alls)))
    w.close()
    return rows


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0] if argv else "docs/figures"); out.mkdir(parents=True, exist_ok=True)
    for f in (ct_radar, ungm, reactor, consistency, lorenz, sysid_fig, static_posterior,
              systems_gallery, static_gallery):
        print(f.__name__); f(out)
    if len(argv) > 1 and Path(argv[1]).exists():
        print("benchmark figures"); benchmark_figs(out, argv[1])
    try:
        print("gui screenshots"); gui_shots(out); gui_data_shots(out)
        for r in gui_latency():
            print("  latency %-16s %-28s first %6.0f ms  all %6.0f ms" % r)
    except ImportError:
        print("  PyQt5 / pyqtgraph missing: skipped")


if __name__ == "__main__":
    main()
