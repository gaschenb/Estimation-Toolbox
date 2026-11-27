"""Static parameter estimation workspace (y = f(θ) + ε): live fits, posteriors, diagnostics."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

from ..core.paramest import METHODS
from ..core.static import list_static
from ..core.userdata import load_static
from . import theme
from .compute import PoolRunner, static_task
from .state_page import Page
from .widgets import (DataSource, Empty, MethodList, ParamEditor, ResultsTable, SciSpin, card, form,
                      log_x, muted, new_plot, no_si, pen, spin, truth_pen)

DESCR = {
    "least-squares": "bounded trust-region least squares, multi-start; s²(JᵀJ)⁻¹ covariance",
    "mle": "joint (θ, σ) maximum likelihood, observed information",
    "map": "MAP with Gaussian prior, Laplace approximation and log-evidence",
    "fisher": "Cramér–Rao bound at the LS estimate, FIM conditioning",
    "profile": "profile-likelihood 95 % intervals",
    "bootstrap": "residual / parametric / wild bootstrap",
    "rwm": "random-walk Metropolis (multi-chain)",
    "am": "adaptive Metropolis (Haario)",
    "ensemble": "affine-invariant ensemble sampler (stretch move)",
    "hmc": "Hamiltonian Monte Carlo, dual averaging, Laplace metric",
}
FAST = {"least-squares", "mle", "map", "fisher"}
LIVE_LIMIT = 0.5


class StaticPage(Page):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.base = self.prob = None
        self.y = None
        self.results, self.runtime, self.stale = {}, {}, set()
        self.live = True
        self.runner = PoolRunner(self)
        self.runner.partial.connect(self._on_partial)
        self.runner.done.connect(self._on_done)
        self.runner.failed.connect(lambda g, m: (self.say("ERROR: " + m),
                                                 self.status.emit("error: " + m.splitlines()[0])))
        self.runner.progressed.connect(lambda g, i, n: (self.bar.setRange(0, n), self.bar.setValue(i)))
        self.runner.busy.connect(lambda b: self.bar.setVisible(b))
        self._debounce = QtCore.QTimer(self); self._debounce.setSingleShot(True)
        self._debounce.setInterval(80); self._debounce.timeout.connect(self._launch)
        self._force = False
        self._build()
        self._problem_changed()

    def _build(self):
        c = self.controls
        f, v = card("Data", "simulate from the model, or load your own: input column(s), y, "
                            "optional sigma")
        self.source = DataSource("static", reader=lambda path: load_static(path, self.base.inputs))
        v.addWidget(self.source)
        c.addWidget(f)
        f, v = card("Problem")
        self.cb = QtWidgets.QComboBox()
        for n, p in list_static().items():
            self.cb.addItem(n, n)
            self.cb.setItemData(self.cb.count() - 1, p.description, Qt.ToolTipRole)
        v.addWidget(self.cb)
        self.lbl_cat = muted(""); v.addWidget(self.lbl_cat)
        self.lbl = muted(""); v.addWidget(self.lbl)
        c.addWidget(f)
        f, v = card("True parameters θ")
        self.ttl_theta = f.findChild(QtWidgets.QLabel, "cardTitle")
        self.lbl_theta = muted(""); self.lbl_theta.setWordWrap(True); v.addWidget(self.lbl_theta)
        self.params = ParamEditor(); v.addWidget(self.params)
        fl = form()
        self.sp_noise = SciSpin(1e-12, 1e12, 0.05); fl.addRow("noise σ", self.sp_noise)
        hs = QtWidgets.QHBoxLayout(); self.sp_seed = spin(0, 10**6, 0)
        bn = QtWidgets.QPushButton("New")
        bn.clicked.connect(lambda: self.sp_seed.setValue(int(np.random.default_rng().integers(10**6))))
        hs.addWidget(self.sp_seed, 1); hs.addWidget(bn); fl.addRow("seed", hs)
        v.addLayout(fl)
        b = QtWidgets.QPushButton("Reset θ"); b.clicked.connect(lambda: self.params.reset())
        v.addWidget(b, 0, Qt.AlignLeft)
        c.addWidget(f)
        f, v = card("Methods", "fast methods update live; samplers run on Run")
        self.methods = MethodList(DESCR); self.methods.fit_height(); v.addWidget(self.methods)
        self.methods.set_checked(["least-squares", "fisher", "hmc"])
        c.addWidget(f)
        f, v = card("Samplers / bootstrap")
        fl = form()
        self.sp_it = spin(200, 10**6, 3000, 500); fl.addRow("iterations", self.sp_it)
        self.sp_burn = spin(50, 10**6, 1000, 100); fl.addRow("burn-in", self.sp_burn)
        self.sp_ch = spin(1, 64, 4); fl.addRow("chains", self.sp_ch)
        self.sp_boot = spin(10, 10**5, 300, 50); fl.addRow("bootstrap B", self.sp_boot)
        self.cb_boot = QtWidgets.QComboBox(); self.cb_boot.addItems(["residual", "parametric", "wild"])
        fl.addRow("bootstrap kind", self.cb_boot)
        v.addLayout(fl)
        c.addWidget(f)
        self.b_run = QtWidgets.QPushButton("Run all methods"); self.b_run.setObjectName("primary")
        self.b_run.clicked.connect(self.run_all); c.addWidget(self.b_run)
        self.bar = QtWidgets.QProgressBar(); self.bar.setVisible(False); c.addWidget(self.bar)
        c.addStretch(1)

        self.v_fit = self.add_view("Fit", self.draw_fit)
        self.v_post = self.add_view("Posterior", self.draw_post)
        self.v_tr = self.add_view("Traces", self.draw_traces)
        self.v_prof = self.add_view("Profile likelihood", self.draw_profile)
        self.v_cmp = self.add_view("Compare", self.draw_compare)
        self.cb_show = QtWidgets.QComboBox()
        self.cb_show.currentIndexChanged.connect(lambda _: self.invalidate(self.v_post, self.v_tr))
        for vw in (self.v_post, self.v_tr):
            pass
        self.v_post.header.addWidget(QtWidgets.QLabel("method")); self.v_post.header.addWidget(self.cb_show)
        self.v_post.header.addWidget(muted("diagonal: marginals · off-diagonal: joint histograms · "
                                           "+ truth"))
        self.v_post.header.addStretch(1)
        self.set_table(ResultsTable(["method", "parameter", "true", "estimate", "stderr", "z",
                                     "interval", "R̂", "ESS bulk", "ESS/s", "time [s]"]))

        self.cb.currentIndexChanged.connect(lambda _: self._problem_changed())
        self.source.changed.connect(self._source_changed)
        self.params.changed.connect(self.schedule)
        self.sp_noise.valueChanged.connect(self.schedule)
        self.sp_seed.valueChanged.connect(self.schedule)
        self.methods.changed.connect(self.schedule)
        for w in (self.sp_it, self.sp_burn, self.sp_ch, self.sp_boot):
            w.valueChanged.connect(self._settings_changed)
        self.cb_boot.currentIndexChanged.connect(self._settings_changed)

    # ------------------------------------------------------------------ problem
    def data(self):
        return self.source.data if self.source.from_file() else None

    def _source_changed(self):
        file = self.source.from_file()
        model = self.cb.model()
        for i in range(self.cb.count()):
            ok = list_static()[self.cb.itemData(i)].accepts_data or not file
            model.item(i).setEnabled(ok)
        if file and not list_static()[self.cb.currentData()].accepts_data:
            ok = [i for i in range(self.cb.count()) if model.item(i).isEnabled()]
            general = [i for i in ok if list_static()[self.cb.itemData(i)].category == "general"]
            self.cb.setCurrentIndex((general or ok)[0])      # triggers _problem_changed
        else:
            self._problem_changed()

    def _problem_changed(self):
        p = list_static()[self.cb.currentData()]
        self.base = p
        file = self.source.from_file()
        if file and not self.source.reload():       # the file lacks this problem's inputs
            self.source.rb_sim.setChecked(True)
            file = False
        d = self.data()
        shown = p.with_data(d)[0] if d is not None else p
        n_obs = len(d.y) if d is not None else p.forward(p.theta_true).size
        self.lbl_cat.setText(f"{p.category} · {p.dim} parameters · {n_obs} observations"
                             + ("" if p.accepts_data else " · fixed design (simulated only)"))
        self.lbl.setText(p.description)
        self.ttl_theta.setText("True parameters θ" if d is None else "Starting values θ₀")
        self.lbl_theta.setText(
            "data are re-simulated live as you drag" if d is None else
            "where the fits start (and the MAP prior mean); defaults are guessed from your data")
        self.params.blockSignals(True)
        self.params.set_params(p.param_names, shown.theta_true, shown.lo, shown.hi)
        self.params.blockSignals(False)
        self.sp_noise.blockSignals(True); self.sp_noise.setValue(shown.noise_std)
        self.sp_noise.blockSignals(False)
        self.sp_noise.setEnabled(d is None or d.sigma is None)
        self.sp_noise.setToolTip("" if d is None else
                                 "per-point σ from the file (residuals are whitened)" if d.sigma
                                 is not None else "estimated from neighbouring points; set it if "
                                                  "you know the measurement noise")
        self.results, self.runtime, self.stale = {}, {}, set()
        self.data_key = None
        self.schedule()

    def _settings_changed(self):
        for m in ("profile", "bootstrap", "rwm", "am", "ensemble", "hmc"):
            self.runtime[m] = max(self.runtime.get(m, 0.0), LIVE_LIMIT + 1)
        self.schedule()

    def schedule(self, *_):
        self._debounce.start()

    def run_all(self):
        self._force = True
        self._launch()

    def cancel(self):
        self.runner.cancel(); self.bar.setVisible(False)

    def set_live(self, on):
        self.live = on
        if on:
            self.schedule()

    def _is_live(self, m):
        return m in FAST if m not in self.runtime else self.runtime[m] <= LIVE_LIMIT

    def _launch(self):
        self._debounce.stop()
        force, self._force = self._force, False
        if not self.live and not force:
            return
        d = self.data()
        noise, seed = self.sp_noise.value(), self.sp_seed.value()
        if d is None:
            th = np.clip(self.params.values(), self.base.lo, self.base.hi)
            key = (self.base.name, tuple(th), noise, seed)
        else:
            th = self.params.values()
            key = ("file", self.source.path, id(d), self.base.name, tuple(th), noise)
        new_data = key != getattr(self, "data_key", None)
        methods = self.methods.checked()
        run = methods if force else [m for m in methods if self._is_live(m)]
        if new_data:
            self.results = {}
        self.results = {m: r for m, r in self.results.items() if m in methods}
        if not force:
            # a re-run of live methods only; keep others if the data did not change
            pass
        self.stale = {m for m in methods if m not in run and m not in self.results}
        kw = dict(n_iter=self.sp_it.value(), burn=min(self.sp_burn.value(), self.sp_it.value() - 50),
                  n_chains=self.sp_ch.value(), n_boot=self.sp_boot.value(),
                  kind=self.cb_boot.currentText())
        base = self.base
        if new_data:
            if d is None:
                self.prob = replace(base, theta_true=th)
                self.y, self.sigma = self.prob.simulate(noise, seed)
            else:
                self.prob, self.y, self.sigma = base.with_data(d, th)
                if d.sigma is None:
                    self.sigma = noise
                    self.prob = replace(self.prob, noise_std=noise)
            self.data_key = key
            self.invalidate(delay=150 if run else 10)
        spec = base.name if d is None else ("data", base.name, d, th)
        tasks = [(static_task, (spec, th, self.y, self.sigma, m, kw, seed)) for m in run]
        self.runner.submit(tasks)
        self.status.emit("estimating: " + (", ".join(run) or "simulating"))
        self._table()

    def _on_partial(self, gen, key, obj):
        if not self.runner.is_current(gen):
            return
        if key == "_data":
            self.data_key, self.prob, self.y, self.sigma = obj
            self.invalidate()
            return
        r, dt = obj
        self.results[key] = r
        self.results = {k: self.results[k] for k in METHODS if k in self.results}
        self.runtime[key] = dt
        self.stale.discard(key)
        cur = self.cb_show.currentText()
        names = [m for m, rr in self.results.items() if rr.samples is not None]
        self.cb_show.blockSignals(True); self.cb_show.clear(); self.cb_show.addItems(names)
        if cur in names:
            self.cb_show.setCurrentText(cur)
        elif key in names:
            self.cb_show.setCurrentText(key)
        self.cb_show.blockSignals(False)
        self._table()
        self.invalidate()

    def _on_done(self, gen, wall):
        if self.runner.is_current(gen) and wall is not None:
            self.status.emit(f"done in {wall:.2f} s wall-clock ({self.runner.mode} pool) — "
                             + ", ".join(f"{m} {self.runtime[m]:.2f} s" for m in self.results
                                         if m in self.runtime))

    def _table(self):
        rows, keys, flags = [], [], []
        if self.prob is None:
            self.table.set_rows([]); return
        for m, r in self.results.items():
            ex = r.extras
            for j, n in enumerate(self.prob.param_names):
                t = self.prob.theta_true[j] if self.prob.has_truth else np.nan
                e, s = r.theta_hat[j], r.stderr[j]
                z = (e - t) / s if s > 0 else np.nan
                ci = f"[{r.ci[j, 0]:.4g}, {r.ci[j, 1]:.4g}]" if r.ci is not None else ""
                rows.append([m if j == 0 else "", n, float(t), float(e), float(s), z, ci,
                             float(ex["rhat"][j]) if "rhat" in ex else np.nan,
                             float(ex["ess_bulk"][j]) if "ess_bulk" in ex else np.nan,
                             float(ex["ess_per_s"][j]) if "ess_per_s" in ex else np.nan,
                             r.runtime if j == 0 else np.nan])
                keys.append(m if j == 0 else None)
                fl = {5: "good" if abs(z) < 2 else "bad"} if np.isfinite(z) else {}
                if "rhat" in ex:
                    fl[7] = "good" if ex["rhat"][j] < 1.01 else "bad"
                flags.append(fl)
        for m in sorted(self.stale):
            rows.append([m, "", np.nan, np.nan, np.nan, np.nan, "press Run", np.nan, np.nan, np.nan,
                         np.nan])
            keys.append(m); flags.append({"row": "stale"})
        self.table.set_rows(rows, keys, flags)

    # ------------------------------------------------------------------ drawing
    def _x(self):
        p = self.prob
        return np.asarray(p.x, float) if p.x is not None else np.arange(self.y.size, dtype=float)

    def draw_fit(self, glw):
        if self.y is None:
            raise Empty("simulating …")
        p, y = self.prob, self.y
        sc = 1.0 if p.scale is None else p.scale         # undo whitening for display
        yt = p.forward(p.theta_true) * sc if p.has_truth else None
        y_show = y * sc
        a = new_plot(glw, 0, legend=True, ylabel="y")
        r_ = new_plot(glw, 1, ylabel="residual" if p.scale is None else "residual / σᵢ",
                      xlabel=p.xlabel if p.plot_kind != "xy" else "index")
        glw.ci.layout.setRowStretchFactor(1, 3); glw.ci.layout.setRowStretchFactor(2, 1)
        fits = {m: p.forward(r.theta_hat) * sc for m, r in self.results.items()}
        if p.plot_kind == "xy":
            d, t = y.reshape(-1, 2), yt.reshape(-1, 2)
            a.plot(d[:, 0], d[:, 1], pen=None, symbol="o", symbolSize=4, symbolPen=None,
                   symbolBrush=pg.mkBrush(theme.current["truth"]), name="data")
            a.plot(t[:, 0], t[:, 1], pen=truth_pen(1.2), name="truth")
            for m, f in fits.items():
                f = f.reshape(-1, 2)
                a.plot(f[:, 0], f[:, 1], pen=pen(m, 1.8), name=m)
            a.setLabel("bottom", "x"); a.setAspectLocked(False)
            xr = np.arange(y.size)
        else:
            x = self._x() if p.plot_kind == "series" else np.arange(y.size, dtype=float)
            if p.xlog and p.plot_kind == "series":
                x = log_x(x, a, r_)
            xr = x
            groups = [(None, np.argsort(x, kind="stable"))] if p.groups is None else \
                [(g, np.flatnonzero(p.groups == g)) for g in dict.fromkeys(p.groups)]
            symbols = ["o", "s", "t", "d", "+", "x"]
            for gi, (g, idx) in enumerate(groups):
                idx = idx[np.argsort(x[idx], kind="stable")]
                a.plot(x[idx], y_show[idx], pen=None, symbol=symbols[gi % 6], symbolSize=5,
                       symbolPen=None, symbolBrush=pg.mkBrush(theme.current["truth"]),
                       name=("data" if g is None else f"data {g}"))
                if yt is not None:
                    a.plot(x[idx], yt[idx], pen=pg.mkPen(theme.current["muted"], width=1.2,
                                                         style=Qt.DashLine),
                           name="truth" if gi == 0 else None)
                for m, f in fits.items():
                    a.plot(x[idx], f[idx], pen=pen(m, 1.8), name=m if gi == 0 else None)
            a.setLabel("bottom", p.xlabel)
        for m, f in fits.items():
            res = (y_show - f) / sc
            r_.plot(xr, res, pen=None, symbol="o", symbolSize=3, symbolPen=None,
                    symbolBrush=pg.mkBrush(theme.qcolor(m, 200)))
        r_.addLine(y=0, pen=pg.mkPen(theme.current["muted"]))
        if self.results:
            s = self.sigma
            r_.addLine(y=2 * s, pen=pg.mkPen(theme.current["muted"], style=Qt.DotLine))
            r_.addLine(y=-2 * s, pen=pg.mkPen(theme.current["muted"], style=Qt.DotLine))
        r_.setXLink(a) if p.plot_kind == "series" else None

    def _sel(self):
        m = self.cb_show.currentText()
        if not m or m not in self.results:
            raise Empty("run a sampler (rwm / am / ensemble / hmc) or the bootstrap")
        return m, self.results[m]

    def draw_post(self, glw):
        m, r = self._sel()
        S = r.samples
        names = self.prob.param_names
        d = len(names)
        lut = theme.heat_lut()
        tr = self.prob.theta_true
        has = self.prob.has_truth
        lims = [np.percentile(S[:, j], [0.5, 99.5]) for j in range(d)]
        if has:
            lims = [(min(a, tr[j]), max(b, tr[j])) for j, (a, b) in enumerate(lims)]
        lims = [(a - 0.06 * (b - a + 1e-12), b + 0.06 * (b - a + 1e-12)) for a, b in lims]
        cols = {}
        for i in range(d):
            for j in range(i + 1):
                p = glw.addPlot(row=i, col=j)
                p.hideButtons(); no_si(p)
                p.getAxis("left").setWidth(48 if j == 0 else 0)
                if j > 0:
                    p.hideAxis("left")
                if i < d - 1:
                    p.getAxis("bottom").setStyle(showValues=False)
                else:
                    p.setLabel("bottom", names[j])
                if j == 0 and i > 0:
                    p.setLabel("left", names[i])
                if i == j:
                    h, e = np.histogram(S[:, i], bins=40, range=lims[i], density=True)
                    p.plot(e, h, stepMode="center", fillLevel=0, pen=pen(m, 1.2),
                           brush=pg.mkBrush(theme.qcolor(m, 90)))
                    if has:
                        p.addLine(x=tr[i], pen=truth_pen(1.4))
                    p.addLine(x=r.theta_hat[i], pen=pen(m, 1.2, Qt.DashLine))
                    if i == 0:
                        p.setLabel("left", names[0])
                else:
                    H, xe, ye = np.histogram2d(S[:, j], S[:, i], bins=36, range=[lims[j], lims[i]])
                    img = pg.ImageItem(np.sqrt(H.T / max(H.max(), 1)))
                    img.setLookupTable(lut)
                    img.setRect(QtCore.QRectF(xe[0], ye[0], xe[-1] - xe[0], ye[-1] - ye[0]))
                    p.addItem(img)
                    if has:
                        p.plot([tr[j]], [tr[i]], pen=None, symbol="+", symbolSize=14,
                               symbolPen=pg.mkPen(theme.current["truth"], width=2))
                p.setXRange(*lims[j], padding=0)
                if i != j:
                    p.setYRange(*lims[i], padding=0)
                cols.setdefault(j, []).append(p)
        for j, ps in cols.items():
            for q in ps[1:]:
                q.setXLink(ps[0])

    def draw_traces(self, glw):
        m, r = self._sel()
        ch = r.extras.get("chains")
        if ch is None:
            ch = r.samples[None]
        names = self.prob.param_names
        p0 = None
        for j, n in enumerate(names):
            p = new_plot(glw, j, ylabel=n, xlabel="iteration" if j == len(names) - 1 else None)
            for c in range(min(len(ch), 8)):
                p.plot(ch[c, :, j], pen=pg.mkPen(pg.intColor(c, hues=max(len(ch), 4), alpha=170),
                                                 width=0.8))
            if self.prob.has_truth:
                p.addLine(y=self.prob.theta_true[j], pen=truth_pen(1.3))
            p0 = p0 or p
            if p is not p0:
                p.setXLink(p0)

    def draw_profile(self, glw):
        r = self.results.get("profile")
        if r is None:
            raise Empty("tick 'profile' and press Run")
        prof, crit = r.extras["profiles"], r.extras["crit"]
        names = self.prob.param_names
        ncol = 2 if len(names) > 1 else 1
        for j, n in enumerate(names):
            g, v = prof[n]
            p = new_plot(glw, j // ncol, j % ncol, xlabel=n, ylabel="2Δℓ" if j % ncol == 0 else None)
            lo, hi = r.ci[j]
            reg = pg.LinearRegionItem(values=(lo, hi), movable=False,
                                      brush=pg.mkBrush(theme.qcolor("profile", 45)), pen=pg.mkPen(None))
            p.addItem(reg)
            p.plot(g, 2 * np.asarray(v), pen=pen("profile", 1.6), symbol="o", symbolSize=4,
                   symbolBrush=pg.mkBrush(theme.color("profile")), symbolPen=None)
            p.addLine(y=2 * crit, pen=pg.mkPen("#dc2626", style=Qt.DotLine))
            if self.prob.has_truth:
                p.addLine(x=self.prob.theta_true[j], pen=truth_pen(1.2))

    def draw_compare(self, glw):
        if not self.results:
            raise Empty("run at least one method")
        names = self.prob.param_names
        ms = list(self.results)
        ncol = 2 if len(names) > 1 else 1
        for j, n in enumerate(names):
            p = new_plot(glw, j // ncol, j % ncol, xlabel=n)
            p.getAxis("left").setTicks([[(i, m) for i, m in enumerate(ms)]])
            p.getAxis("left").setWidth(80)
            for i, m in enumerate(ms):
                r = self.results[m]
                if r.ci is not None:
                    p.plot(r.ci[j], [i, i], pen=pen(m, 6, alpha=90))
                e = 1.645 * r.stderr[j]
                p.addItem(pg.ErrorBarItem(x=np.array([r.theta_hat[j]]), y=np.array([i]),
                                          left=np.array([e]), right=np.array([e]), beam=0.2,
                                          pen=pen(m, 1.6)))
                p.plot([r.theta_hat[j]], [i], pen=None, symbol="o", symbolSize=7, symbolPen=None,
                       symbolBrush=pg.mkBrush(theme.color(m)))
            if self.prob.has_truth:
                p.addLine(x=self.prob.theta_true[j], pen=truth_pen(1.2))
            p.setYRange(-0.7, len(ms) - 0.3)
