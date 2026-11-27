"""Matplotlib figures shared by the CLI, the gallery and the GUI.
Every function draws into a given Figure (so the GUI can reuse its canvases)."""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

PALETTE = {
    "kf": "#1f77b4", "srkf": "#17becf", "ekf": "#ff7f0e", "iekf": "#bcbd22", "ukf": "#2ca02c",
    "ckf": "#9467bd", "ghkf": "#8c564b", "iplf": "#e377c2", "enkf": "#7f7f7f", "etkf": "#393b79",
    "pf": "#d62728", "rpf": "#ff9896", "pf-ekf": "#ad494a", "mhe": "#637939", "batch-map": "#000000",
    "least-squares": "#1f77b4", "mle": "#ff7f0e", "map": "#2ca02c", "fisher": "#8c564b",
    "profile": "#17becf", "bootstrap": "#9467bd", "rwm": "#7f7f7f", "am": "#bcbd22",
    "ensemble": "#e377c2", "hmc": "#d62728",
}


def color(name):
    base = name.split("+")[0]
    return PALETTE.get(base, "#444444")


def _grid(fig, n, ncols=None):
    ncols = ncols or (1 if n <= 3 else 2)
    nrows = int(np.ceil(n / ncols))
    return [fig.add_subplot(nrows, ncols, i + 1) for i in range(n)]


# ----------------------------------------------------------------- state estimation
def plot_states(fig, model, xs, results, states=None, band=True, t=None):
    states = list(range(model.nx)) if states is None else states
    axes = _grid(fig, len(states))
    t = np.arange(len(xs)) * model.dt if t is None else t
    for ax, i in zip(axes, states):
        ax.plot(t, xs[:, i], "k-", lw=1.6, label="truth")
        for name, r in results.items():
            if r is None or not np.all(np.isfinite(r.m[:, i])):
                continue
            c = color(name)
            ls = "--" if name.endswith("+s") or "+" in name else "-"
            ax.plot(t, r.m[:, i], ls, color=c, lw=1.1, label=name)
            if band:
                sd = np.sqrt(np.maximum(r.P[:, i, i], 0))
                ax.fill_between(t, r.m[:, i] - 2 * sd, r.m[:, i] + 2 * sd, color=c, alpha=0.12,
                                lw=0)
        ax.set_ylabel(model.state_names[i])
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time")
    axes[0].legend(fontsize=7, ncol=3, loc="best")
    fig.suptitle(f"{model.name}: estimates ± 2σ")
    return axes


def plot_measurements(fig, model, xs, ys, t=None):
    axes = _grid(fig, model.ny)
    t = np.arange(1, len(ys) + 1) * model.dt if t is None else t
    yt = model.h(xs[1:], np.arange(1, len(xs))[:, None]) if model.gaussian_obs else None
    for j, ax in enumerate(axes):
        ax.plot(t, ys[:, j], ".", ms=3, color="0.4", label="measurement")
        if yt is not None:
            ax.plot(t, yt[:, j], "k-", lw=1, label="h(truth)")
        ax.set_ylabel(model.obs_names[j]); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[-1].set_xlabel("time")
    fig.suptitle(f"{model.name}: measurements")


def plot_track(fig, model, xs, results, idx=(0, 1), ys=None, problem=None):
    ax = fig.add_subplot(111)
    i, j = idx
    ax.plot(xs[:, i], xs[:, j], "k-", lw=2, label="truth")
    if problem is not None:
        if ys is not None and problem.meas_xy is not None:
            m = problem.meas_xy(ys)
            ax.plot(m[:, 0], m[:, 1], ".", ms=3, color="0.6", label="measurements")
        for n, pts in problem.landmarks.items():
            pts = np.asarray(pts)
            if len(pts) > 20:
                ax.plot(pts[:, 0], pts[:, 1], ":", color="0.3", lw=1.2, label=n)
            else:
                ax.plot(pts[:, 0], pts[:, 1], "k^", ms=8, label=n)
    for name, r in results.items():
        if r is None or not np.all(np.isfinite(r.m)):
            continue
        ax.plot(r.m[:, i], r.m[:, j], "--" if "+" in name else "-", color=color(name), lw=1.2,
                label=name)
    ax.set_xlabel(model.state_names[i]); ax.set_ylabel(model.state_names[j])
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax.set_title(f"{model.name}: track")


def plot_errors(fig, model, xs, results, states=None):
    states = list(range(model.nx)) if states is None else states
    axes = _grid(fig, len(states))
    t = np.arange(len(xs)) * model.dt
    for ax, i in zip(axes, states):
        for name, r in results.items():
            if r is None or not np.all(np.isfinite(r.m)):
                continue
            c = color(name)
            sd = np.sqrt(np.maximum(r.P[:, i, i], 0))
            ax.plot(t, xs[:, i] - r.m[:, i], color=c, lw=1, label=name)
            ax.plot(t, 2 * sd, ":", color=c, lw=0.9); ax.plot(t, -2 * sd, ":", color=c, lw=0.9)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel(f"error {model.state_names[i]}"); ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time")
    axes[0].legend(fontsize=7, ncol=3)
    fig.suptitle("estimation error and ±2σ envelope")


def plot_consistency(fig, nees, nis, nx, ny, n_runs=1, alpha=0.05):
    """nees/nis: {method: (T,) or (R, T) arrays}. Draws run-averaged NEES/NIS with the
    two-sided χ² band for the average of n_runs variables."""
    ax1, ax2 = fig.add_subplot(211), fig.add_subplot(212)
    for ax, data, dof, lab in ((ax1, nees, nx, "NEES"), (ax2, nis, ny, "NIS")):
        Tm = 1
        for name, a in data.items():
            a = np.atleast_2d(a)
            if not np.any(np.isfinite(a)):
                continue
            m = np.nanmean(a, 0)
            Tm = max(Tm, len(m))
            ax.semilogy(np.arange(1, len(m) + 1), m, "--" if "+" in name else "-",
                        color=color(name), lw=1, label=name)
        R = n_runs
        lo, hi = chi2.ppf([alpha / 2, 1 - alpha / 2], R * dof) / R
        ax.axhspan(lo, hi, color="g", alpha=0.12, label=f"{100 * (1 - alpha):.0f}% band")
        ax.axhline(dof, color="g", lw=0.8)
        ax.set_ylabel(("average " if R > 1 else "") + lab); ax.grid(alpha=0.3, which="both")
        ax.set_xlim(1, Tm)
    ax1.legend(fontsize=6, ncol=6, loc="lower right", framealpha=0.7)
    ax2.set_xlabel("time step")
    fig.suptitle(f"consistency: χ²({nx}) NEES and χ²({ny}) NIS"
                 + (f", {n_runs} Monte Carlo runs" if n_runs > 1 else ", single run"))


def plot_innovations(fig, results, max_lag=25):
    from .consistency import autocorr, whiten
    ax1, ax2 = fig.add_subplot(211), fig.add_subplot(212)
    for name, r in results.items():
        v, S = getattr(r, "v", None), getattr(r, "S", None)
        if v is None or S is None or not np.all(np.isfinite(S)):
            continue
        z = whiten(v, S)
        ax1.plot(np.arange(1, len(z) + 1), z[:, 0], color=color(name), lw=0.8, label=name)
        ac, b = autocorr(z, min(max_lag, len(z) // 3))
        ax2.plot(np.arange(1, len(ac) + 1), ac, "o-", ms=3, color=color(name), lw=0.8)
    ax1.axhspan(-1.96, 1.96, color="g", alpha=0.1)
    ax1.set_ylabel("whitened innovation (1st comp.)"); ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7, ncol=4)
    if ax2.lines:
        T = len(z)
        bb = 1.96 / np.sqrt(T * z.shape[1])
        ax2.axhspan(-bb, bb, color="g", alpha=0.12)
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xlabel("lag"); ax2.set_ylabel("autocorrelation"); ax2.grid(alpha=0.3)
    fig.suptitle("innovation whiteness")


def plot_particles(fig, model, xs, pf, comp=0, smoothed=None, bins=80):
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    X, LW = pf.particles[:, :, comp], pf.logw
    lo, hi = np.nanpercentile(X, 0.5), np.nanpercentile(X, 99.5)
    pad = 0.05 * (hi - lo + 1e-9)
    edges = np.linspace(lo - pad, hi + pad, bins + 1)
    H = np.zeros((bins, len(X)))
    for k in range(len(X)):
        H[:, k] = np.histogram(X[k], edges, weights=np.exp(LW[k]))[0]
    t = np.arange(len(X)) * model.dt
    ax1.imshow(np.sqrt(H), origin="lower", aspect="auto", cmap="Blues",
               extent=[t[0] - 0.5 * model.dt, t[-1] + 0.5 * model.dt, edges[0], edges[-1]])
    if smoothed is not None and smoothed.samples is not None:
        for i, tr in enumerate(smoothed.samples[:20]):
            ax1.plot(t, tr[:, comp], color="0.45", lw=0.4, alpha=0.6,
                     label="FFBSi draws" if i == 0 else None)
    ax1.plot(t, xs[:, comp], "k-", lw=1.3, label="truth")
    ax1.plot(t, pf.m[:, comp], "--", color="tab:red", lw=1.0, label="PF mean")
    ax1.set_ylabel(model.state_names[comp]); ax1.legend(fontsize=7, loc="upper right", ncol=3)
    ax1.set_title("particle filtering density (√ of the weighted histogram per step)")
    ax2.plot(np.arange(1, len(pf.ess) + 1) * model.dt, pf.ess, "k-", lw=0.8)
    ax2.axhline(pf.particles.shape[1] * 0.5, color="r", lw=0.6, ls=":")
    ax2.set_ylabel("ESS before resampling"); ax2.set_xlabel("time"); ax2.grid(alpha=0.3)


def plot_mc_summary(fig, summary, title=""):
    names = [m for m in summary if summary[m].get("n_ok", 0) > 0]
    axes = [fig.add_subplot(1, 3, i + 1) for i in range(3)]
    x = np.arange(len(names))
    cols = [color(n) for n in names]
    axes[0].bar(x, [summary[n]["rmse"] for n in names], color=cols)
    axes[0].set_title("RMSE")
    an = [summary[n].get("anees_median", np.nan) for n in names]
    axes[1].bar(x, an, color=cols); axes[1].set_yscale("log")
    lo, hi = summary[names[0]]["anees_band"]
    axes[1].axhspan(lo, hi, color="g", alpha=0.15)
    axes[1].set_title("ANEES (median over time)")
    axes[2].bar(x, [summary[n]["time_ms"] for n in names], color=cols)
    axes[2].set_yscale("log"); axes[2].set_title("runtime per run [ms]")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=60, fontsize=7); ax.grid(alpha=0.3, axis="y")
    for i, n in enumerate(names):
        if summary[n].get("n_diverged", 0):
            axes[0].text(i, summary[n]["rmse"], f"{summary[n]['n_diverged']}✗", ha="center",
                         va="bottom", fontsize=7, color="r")
    fig.suptitle(title)


# ------------------------------------------------------------- static estimation
def _static_x(prob):
    if prob.x is not None:
        return np.asarray(prob.x, float)
    return np.arange(prob.forward(prob.theta_true).size, dtype=float)


def _static_curves(prob):
    """[(label, index array sorted by x)] — one curve per group."""
    x = _static_x(prob)
    if prob.groups is None:
        return [(None, np.argsort(x, kind="stable"))]
    out = []
    for g in dict.fromkeys(prob.groups):
        idx = np.flatnonzero(prob.groups == g)
        out.append((g, idx[np.argsort(x[idx], kind="stable")]))
    return out


def plot_static_fit(fig, prob, y, results):
    ax = fig.add_subplot(211); axr = fig.add_subplot(212)
    sc = 1.0 if getattr(prob, "scale", None) is None else prob.scale   # undo whitening
    has = getattr(prob, "has_truth", True)
    yt = prob.forward(prob.theta_true) * sc
    y = y * sc
    if prob.plot_kind == "xy":
        d, t = y.reshape(-1, 2), yt.reshape(-1, 2)
        ax.plot(d[:, 0], d[:, 1], "k.", ms=3, label="data")
        ax.plot(t[:, 0], t[:, 1], "g--", lw=1, label="truth")
        for n, r in results.items():
            h = prob.forward(r.theta_hat).reshape(-1, 2)
            ax.plot(h[:, 0], h[:, 1], color=color(n), lw=1.1, label=n)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        xr = np.arange(y.size)
    else:
        x = xr = _static_x(prob) if prob.plot_kind == "series" else np.arange(y.size, dtype=float)
        for c, (g, idx) in enumerate(_static_curves(prob) if prob.plot_kind == "series"
                                     else [(None, np.arange(y.size))]):
            first = c == 0
            ax.plot(x[idx], y[idx], ".", color="k" if g is None else f"C{c}", ms=3,
                    label="data" if first and g is None else g)
            if has:
                ax.plot(x[idx], yt[idx], "g--", lw=1, label="truth" if first else None)
            for n, r in results.items():
                o = idx[np.argsort(x[idx], kind="stable")]
                ax.plot(x[o], (prob.forward(r.theta_hat) * sc)[o], color=color(n), lw=1.1,
                        label=n if first else None)
        ax.set_xlabel(prob.xlabel if prob.plot_kind == "series" else "observation index")
        if prob.xlog and prob.plot_kind == "series":
            ax.set_xscale("log")
    ax.legend(fontsize=7, ncol=4); ax.grid(alpha=0.3)
    for n, r in results.items():
        res = (y - prob.forward(r.theta_hat) * sc) / sc
        axr.plot(xr, res, ".", ms=2, color=color(n), label=f"{n} (rms {np.sqrt(np.mean(res**2)):.3g})")
    if prob.xlog and prob.plot_kind == "series":
        axr.set_xscale("log")
    axr.axhline(0, color="k", lw=0.6); axr.set_ylabel("residual"); axr.grid(alpha=0.3)
    if results:
        axr.legend(fontsize=7, ncol=3)
    fig.suptitle(f"{prob.name}: fit")


def plot_corner(fig, names, samples, truth=None, estimate=None, bins=40, c="tab:blue"):
    d = samples.shape[1]
    for i in range(d):
        for j in range(i + 1):
            ax = fig.add_subplot(d, d, i * d + j + 1)
            if i == j:
                ax.hist(samples[:, i], bins=bins, density=True, color=c, alpha=0.7)
                if truth is not None:
                    ax.axvline(truth[i], color="g", ls="--")
                if estimate is not None:
                    ax.axvline(estimate[i], color="r", lw=0.8)
                ax.set_yticks([])
            else:
                ax.hist2d(samples[:, j], samples[:, i], bins=bins, cmap="Blues")
                if truth is not None:
                    ax.plot(truth[j], truth[i], "g+", ms=10, mew=2)
            if i == d - 1:
                ax.set_xlabel(names[j], fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(names[i], fontsize=8)
            elif j > 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=6)


def plot_traces(fig, names, chains, truth=None):
    d = chains.shape[-1]
    axes = _grid(fig, d, 1)
    for j, ax in enumerate(axes):
        for c in chains:
            ax.plot(c[:, j], lw=0.4, alpha=0.8)
        if truth is not None:
            ax.axhline(truth[j], color="k", ls="--", lw=0.8)
        ax.set_ylabel(names[j], fontsize=8)
    axes[-1].set_xlabel("iteration (post burn-in)")


def plot_profiles(fig, prob, res):
    prof = res.extras["profiles"]
    axes = _grid(fig, len(prof), 2 if len(prof) > 1 else 1)
    for ax, (j, name) in zip(axes, enumerate(prob.param_names)):
        g, v = prof[name]
        ax.plot(g, 2 * v, "o-", ms=3)
        ax.axhline(2 * res.extras["crit"], color="r", ls=":", label="χ²₁(0.95)")
        ax.axvspan(*res.ci[j], color="tab:blue", alpha=0.12)
        ax.axvline(prob.theta_true[j], color="g", ls="--", label="truth")
        ax.set_xlabel(name); ax.set_ylabel("2Δℓ"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("profile likelihood (95 % intervals shaded)")


def plot_compare(fig, names, truth, results):
    d = len(names)
    axes = _grid(fig, d, 2 if d > 1 else 1)
    for ax, j in zip(axes, range(d)):
        ms = list(results)
        for i, m in enumerate(ms):
            r = results[m]
            if r.ci is not None:
                lo, hi = r.ci[j]
                ax.plot([lo, hi], [i, i], color=color(m), lw=3, alpha=0.5)
            ax.errorbar(r.theta_hat[j], i, xerr=1.645 * r.stderr[j], fmt="o", color=color(m),
                        capsize=3)
        if truth is not None:
            ax.axvline(truth[j], color="g", ls="--")
        ax.set_yticks(range(len(ms))); ax.set_yticklabels(ms, fontsize=7)
        ax.set_title(names[j], fontsize=9); ax.grid(alpha=0.3, axis="x")
    fig.suptitle("estimates: ±1.645σ (bars), 90 %/95 % intervals (shaded)")
