"""System identification workspace: PEM / EM / PMMH on any state-space family."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

from ..core import sysid
from ..core.systems import list_ssm
from ..core.userdata import data_problems, full_truth
from . import theme
from .state_page import Page
from .widgets import (DataSource, Empty, ParamEditor, ResultsTable, Runner, card, form, muted, new_plot, no_si, pen,
                      spin, truth_pen)


class SysIdPage(Page):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.problem = self.res = self.xs = self.ys = None
        self.slices = None
        self.runner = Runner(self)
        self.runner.partial.connect(self._on_partial)
        self.runner.done.connect(self._on_done)
        self.runner.failed.connect(lambda g, m: (self.say("ERROR: " + m),
                                                 self.status.emit("error: " + m.splitlines()[0]),
                                                 self._idle()))
        self.runner.progressed.connect(lambda g, i, n: (self.bar.setRange(0, n), self.bar.setValue(i)))
        self.runner.busy.connect(lambda b: (self.bar.setVisible(b), self.b_run.setText(
            "Cancel" if b else "Estimate parameters")))
        self._build()
        self._system_changed()

    def _build(self):
        c = self.controls
        f, v = card("Data", "simulate, or load measurements (columns t, measurements)")
        self.source = DataSource("series"); v.addWidget(self.source)
        c.addWidget(f)
        f, v = card("Model")
        self.cb = QtWidgets.QComboBox()
        self.catalog = {}
        self._fill(list_ssm())
        v.addWidget(self.cb)
        self.lbl = muted(""); v.addWidget(self.lbl)
        fl = form()
        self.sp_T = spin(10, 20000, 200, 50); fl.addRow("steps T", self.sp_T)
        self.sp_seed = spin(0, 10**6, 0); fl.addRow("seed", self.sp_seed)
        v.addLayout(fl)
        c.addWidget(f)
        f, v = card("True θ (data generation)")
        self.ttl_theta = f.findChild(QtWidgets.QLabel, "cardTitle")
        self.params = ParamEditor(); v.addWidget(self.params)
        c.addWidget(f)
        f, v = card("Estimator")
        fl = form()
        self.cb_m = QtWidgets.QComboBox(); self.cb_m.addItems(["pem", "pem (MAP)", "em", "pmmh"])
        self.cb_m.setItemData(0, "prediction-error / maximum likelihood with a Gaussian filter", Qt.ToolTipRole)
        self.cb_m.setItemData(2, "EM (Shumway–Stoffer) — linear-Gaussian cv-tracking only", Qt.ToolTipRole)
        self.cb_m.setItemData(3, "particle marginal Metropolis–Hastings (exact-approximate)", Qt.ToolTipRole)
        fl.addRow("method", self.cb_m)
        self.cb_f = QtWidgets.QComboBox(); self.cb_f.addItems(["auto", "kf", "ekf", "ukf", "ckf"])
        fl.addRow("PEM filter", self.cb_f)
        self.lst_free = QtWidgets.QListWidget(); self.lst_free.setFixedHeight(96)
        fl.addRow("estimate", self.lst_free)
        self.sp_np = spin(20, 20000, 300, 50); fl.addRow("PMMH particles", self.sp_np)
        self.sp_it = spin(100, 200000, 3000, 500); fl.addRow("PMMH iterations", self.sp_it)
        self.sp_burn = spin(0, 100000, 1000, 100); fl.addRow("PMMH burn-in", self.sp_burn)
        v.addLayout(fl)
        c.addWidget(f)
        self.b_run = QtWidgets.QPushButton("Estimate parameters"); self.b_run.setObjectName("primary")
        self.b_run.clicked.connect(self.run_all); c.addWidget(self.b_run)
        self.bar = QtWidgets.QProgressBar(); self.bar.setVisible(False); c.addWidget(self.bar)
        c.addStretch(1)
        self.v_data = self.add_view("Data", self.draw_data)
        self.v_conv = self.add_view("Convergence", self.draw_conv)
        self.v_post = self.add_view("Posterior", self.draw_post)
        self.v_sl = self.add_view("Likelihood slices", self.draw_slices)
        self.set_table(ResultsTable(["parameter", "true", "estimate", "stderr", "z", "90 % interval"]))
        self.cb.currentIndexChanged.connect(lambda _: self._system_changed())
        self.source.changed.connect(self._source_changed)
        self.params.changed.connect(self._simulate)
        self.sp_T.valueChanged.connect(self._simulate)
        self.sp_seed.valueChanged.connect(self._simulate)

    def _fill(self, catalog):
        self.catalog = dict(catalog)
        self.cb.blockSignals(True); self.cb.clear()
        for n, p in self.catalog.items():
            self.cb.addItem(n, n)
            self.cb.setItemData(self.cb.count() - 1, p.description, Qt.ToolTipRole)
        self.cb.blockSignals(False)

    def data(self):
        return self.source.data if self.source.from_file() else None

    def _source_changed(self):
        d = self.data()
        self._fill(data_problems(d) if d is not None else list_ssm())
        self.cb.setCurrentIndex(0)
        for w in (self.sp_T, self.sp_seed):
            w.setEnabled(d is None)
        self._system_changed()

    def _system_changed(self):
        p = self.catalog[self.cb.currentData()]
        self.problem = p
        md = p.model()
        self.lbl.setText(p.description)
        self.ttl_theta.setText("True θ (data generation)" if self.data() is None else
                               "Starting values θ₀ (prior mean)")
        self.params.blockSignals(True); self.params.set_params(p.param_names, p.theta_true, p.lo, p.hi)
        self.params.blockSignals(False)
        self.sp_T.blockSignals(True); self.sp_T.setValue(max(p.T, 100)); self.sp_T.blockSignals(False)
        self.lst_free.clear()
        for n in p.param_names:
            it = QtWidgets.QListWidgetItem(n)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable); it.setCheckState(Qt.Checked)
            self.lst_free.addItem(it)
        lin = md.linear and list(p.param_names) == ["q", "r"]     # EM: Q = q·Q₁, R = r²I
        self.cb_m.model().item(2).setEnabled(lin)
        for i in (0, 1):
            self.cb_m.model().item(i).setEnabled(md.gaussian_obs)
        self.cb_m.setCurrentIndex(2 if lin else (0 if md.gaussian_obs else 3))
        self._simulate()

    def theta(self):
        return np.clip(self.params.values(), self.problem.lo, self.problem.hi)

    def _simulate(self):
        md = self.problem.model(self.theta())
        d = self.data()
        if d is None:
            self.xs, self.ys = md.simulate(self.sp_T.value(), self.sp_seed.value())
        else:
            self.ys = d.ys
            self.xs = full_truth(d, md)
        self.res = self.slices = None
        self.table.set_rows([])
        self.invalidate()

    def cancel(self):
        self.runner.cancel(); self._idle()

    def _idle(self):
        self.bar.setVisible(False); self.b_run.setText("Estimate parameters")

    def set_live(self, on):
        pass

    def run_all(self):
        if self.runner.running():
            self.cancel(); return
        p, ys, th = self.problem, self.ys, self.theta()
        m = self.cb_m.currentText()
        filt = None if self.cb_f.currentText() == "auto" else self.cb_f.currentText()
        free = [i for i in range(self.lst_free.count())
                if self.lst_free.item(i).checkState() == Qt.Checked] or None
        npart, nit, nb, seed = self.sp_np.value(), self.sp_it.value(), self.sp_burn.value(), \
            self.sp_seed.value()
        from dataclasses import replace
        prob = replace(p, theta_true=th) if self.data() is None else \
            replace(p, theta_true=th, prior_mean=th)

        def job(ctx):
            if m == "em":
                r = sysid.em(prob, ys, 300)
            elif m.startswith("pem"):
                r = sysid.pem(prob, ys, filt, use_prior="MAP" in m, free=free)
            else:
                r = sysid.pmmh(prob, ys, npart, nit, min(nb, nit - 10), rng=seed + 7, free=free,
                               progress=ctx.progress)
            ctx.emit("result", r)
            if prob.model().gaussian_obs:            # likelihood slices through the estimate
                f2 = "kf" if prob.model().linear else "ukf"
                out = {}
                for j in range(len(prob.param_names)):
                    ctx.check()
                    c = r.theta_hat[j]
                    s = r.stderr[j] if np.isfinite(r.stderr[j]) and r.stderr[j] > 0 else 0.1 * abs(c) + 1e-3
                    g = np.unique(np.clip(np.linspace(c - 4 * s, c + 4 * s, 21), prob.lo[j], prob.hi[j]))
                    ll = []
                    for v in g:
                        t = r.theta_hat.copy(); t[j] = v
                        ll.append(sysid.filter_loglik(prob, t, ys, f2))
                    out[prob.param_names[j]] = (g, np.array(ll))
                ctx.emit("slices", (f2, out))
            return r

        self.res = self.slices = None
        self.runner.submit(job)
        self.status.emit(f"{m} on {p.name} …")

    def _on_partial(self, gen, key, obj):
        if not self.runner.is_current(gen):
            return
        if key == "result":
            self.res = obj
            p, r = self.problem, obj
            th = self.theta() if self.data() is None else np.full(len(p.param_names), np.nan)
            rows, flags = [], []
            for j, n in enumerate(p.param_names):
                s = r.stderr[j]
                z = (r.theta_hat[j] - th[j]) / s if s and np.isfinite(s) and s > 0 else np.nan
                ci = f"[{r.ci[j, 0]:.4g}, {r.ci[j, 1]:.4g}]" if r.ci is not None else ""
                rows.append([n, float(th[j]), float(r.theta_hat[j]), float(s), z, ci])
                flags.append({4: "good" if abs(z) < 2 else "bad"} if np.isfinite(z) else {})
            self.table.set_rows(rows, flags=flags)
            self.say(f"{r.method}: log-lik {r.loglik:.3f}, {r.runtime:.2f} s; "
                     + ", ".join(f"{k} = {v:.3g}" for k, v in r.extras.items() if np.isscalar(v)))
            self.status.emit(f"{r.method}: log-likelihood {r.loglik:.3f} in {r.runtime:.2f} s")
        elif key == "slices":
            self.slices = obj
        self.invalidate()

    def _on_done(self, gen, r):
        self._idle()

    # ------------------------------------------------------------------ drawing
    def draw_data(self, glw):
        md = self.problem.model(self.theta())
        d = self.data()
        t = d.t if d is not None else np.arange(1, len(self.ys) + 1) * md.dt
        ny = min(md.ny, 6)
        yt = None
        if self.xs is not None and np.all(np.isfinite(self.xs[1:])):
            yt = md.h(self.xs[1:], np.arange(1, len(self.xs))[:, None])
        p0 = None
        for j in range(ny):
            p = new_plot(glw, j, ylabel=md.obs_names[j], legend=j == 0,
                         xlabel="time" if j == ny - 1 else None)
            p.plot(t, self.ys[:, j], pen=None, symbol="o", symbolSize=3, symbolPen=None,
                   symbolBrush=pg.mkBrush(theme.current["band"]), name="measurement")
            if yt is not None:
                p.plot(t, yt[:, j], pen=truth_pen(1.2), name="h(truth)")
            p0 = p0 or p
            if p is not p0:
                p.setXLink(p0)

    def draw_conv(self, glw):
        if self.res is None:
            raise Empty("press Estimate parameters")
        h = np.asarray(self.res.history, float)
        m = self.res.method
        p = new_plot(glw, 0, title=m,
                     xlabel="iteration (post burn-in)" if m == "pmmh" else "iteration",
                     ylabel="PF log-likelihood estimate" if m == "pmmh" else "log-likelihood")
        if m == "pmmh":
            p.plot(h, pen=pen("pmmh", 0.8))
        else:
            p.plot(np.arange(1, len(h) + 1), h, pen=pen(m.split()[0], 1.6), symbol="o", symbolSize=5,
                   symbolBrush=pg.mkBrush(theme.color(m.split()[0])), symbolPen=None)
        if self.res.samples is not None:
            th = self.theta()
            free = self.res.extras.get("free", list(range(len(th))))
            for row, j in enumerate(free, start=1):
                q = new_plot(glw, row, ylabel=self.problem.param_names[j])
                q.plot(self.res.samples[:, j], pen=pen("pmmh", 0.7))
                if self.data() is None:
                    q.addLine(y=th[j], pen=truth_pen(1.2))
                q.setXLink(p)

    def draw_post(self, glw):
        r = self.res
        if r is None or r.samples is None:
            raise Empty("PMMH gives posterior samples")
        free = r.extras["free"]
        names = [self.problem.param_names[i] for i in free]
        S = r.samples[:, free]
        tr = self.theta()[free]
        d = len(free)
        lut = theme.heat_lut()
        for i in range(d):
            for j in range(i + 1):
                p = glw.addPlot(row=i, col=j)
                no_si(p)
                if i == d - 1:
                    p.setLabel("bottom", names[j])
                if j == 0:
                    p.setLabel("left", names[i])
                if i == j:
                    h, e = np.histogram(S[:, i], bins=40, density=True)
                    p.plot(e, h, stepMode="center", fillLevel=0, pen=pen("pmmh", 1.2),
                           brush=pg.mkBrush(theme.qcolor("pmmh", 90)))
                    if self.data() is None:
                        p.addLine(x=tr[i], pen=truth_pen(1.4))
                else:
                    H, xe, ye = np.histogram2d(S[:, j], S[:, i], bins=32)
                    img = pg.ImageItem(np.sqrt(H.T / max(H.max(), 1)))
                    img.setLookupTable(lut)
                    img.setRect(QtCore.QRectF(xe[0], ye[0], xe[-1] - xe[0], ye[-1] - ye[0]))
                    p.addItem(img)
                    if self.data() is None:
                        p.plot([tr[j]], [tr[i]], pen=None, symbol="+", symbolSize=14,
                               symbolPen=pg.mkPen(theme.current["truth"], width=2))

    def draw_slices(self, glw):
        if self.slices is None:
            raise Empty("likelihood slices appear after a fit (Gaussian-likelihood systems)")
        f2, sl = self.slices
        th = self.theta()
        names = list(sl)
        ncol = 2 if len(names) > 1 else 1
        for j, n in enumerate(names):
            g, ll = sl[n]
            p = new_plot(glw, j // ncol, j % ncol, xlabel=n,
                         ylabel=f"{f2} log-lik" if j % ncol == 0 else None)
            p.plot(g, ll, pen=pen("pem", 1.6), symbol="o", symbolSize=4, symbolPen=None,
                   symbolBrush=pg.mkBrush(theme.color("pem")))
            if self.data() is None:
                p.addLine(x=th[j], pen=truth_pen(1.2))
            p.addLine(x=self.res.theta_hat[j], pen=pg.mkPen("#dc2626", width=1.2))
