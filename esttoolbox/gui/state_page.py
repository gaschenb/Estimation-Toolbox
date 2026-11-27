"""State estimation workspace: live filtering/smoothing with a linked time cursor."""
from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from scipy.stats import chi2

from ..core import consistency as cs
from ..core.run import STATE_METHODS, applicable, merge_mc
from ..core.systems import list_ssm
from ..core.userdata import data_problems, full_truth, problem_spec
from . import theme
from .compute import PoolRunner, mc_task, state_task
from .widgets import (DataSource, Empty, MethodList, ParamEditor, PlotView, ResultsTable, band, bar_ticks,
                      card, ellipse, form, hband, method_style, muted, new_plot, pen, spin, truth_pen)

LIVE_LIMIT = 0.6      # s: methods slower than this are not re-run automatically in live mode
MAX_PANELS = 6


class Page(QtWidgets.QWidget):
    """Controls column | view tabs over a results table + log."""
    status = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("page")
        outer = QtWidgets.QHBoxLayout(self); outer.setContentsMargins(10, 10, 10, 10)
        split = QtWidgets.QSplitter(Qt.Horizontal); split.setChildrenCollapsible(False)
        outer.addWidget(split)
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        holder = QtWidgets.QWidget()
        self.controls = QtWidgets.QVBoxLayout(holder)
        self.controls.setContentsMargins(0, 0, 6, 0); self.controls.setSpacing(10)
        scroll.setWidget(holder); scroll.setMinimumWidth(360); scroll.setMaximumWidth(440)
        split.addWidget(scroll)
        right = QtWidgets.QSplitter(Qt.Vertical); right.setChildrenCollapsible(False)
        self.tabs = QtWidgets.QTabWidget(); self.tabs.setDocumentMode(False)
        self.tabs.currentChanged.connect(lambda _: self.refresh())
        right.addWidget(self.tabs)
        bottom = QtWidgets.QTabWidget()
        self.table = None
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        self.bottom = bottom
        right.addWidget(bottom)
        right.setStretchFactor(0, 4); right.setStretchFactor(1, 1)
        right.setSizes([620, 210])
        split.addWidget(right); split.setStretchFactor(1, 1); split.setSizes([370, 1100])
        self.views = []
        self._redraw_timer = QtCore.QTimer(self); self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(30); self._redraw_timer.timeout.connect(self._do_refresh)

    def set_table(self, table):
        self.table = table
        self.bottom.addTab(table, "Results")
        self.bottom.addTab(self.log, "Log")

    def add_view(self, title, draw, reusable=False):
        v = PlotView(title, draw, reusable=reusable)
        self.tabs.addTab(v, title)
        self.views.append(v)
        return v

    def invalidate(self, *views, delay=25):
        for v in views or self.views:
            v.dirty = True
        self.refresh(delay)

    def refresh(self, delay=25):
        """Coalesced redraw of the visible view. A pending redraw is only ever brought
        forward, so a burst of results produces one repaint."""
        t = self._redraw_timer
        if t.isActive() and t.remainingTime() <= delay:
            return
        t.start(delay)

    def _do_refresh(self):
        w = self.tabs.currentWidget()
        if isinstance(w, PlotView) and w.dirty:
            w.redraw()
        if getattr(self, "_table_dirty", False):
            self._table_dirty = False
            self._refresh_table()

    def say(self, msg):
        self.log.appendPlainText(msg)

    def current_view(self):
        return self.tabs.currentWidget()

    # toolbar hooks
    def run_all(self):
        pass

    def cancel(self):
        pass

    def export_png(self, path):
        w = self.current_view()
        (w.glw if isinstance(w, PlotView) else w).grab().save(path)

    def export_csv(self, path):
        self.table.to_csv(path)


# =========================================================================== state estimation
class StatePage(Page):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.problem = self.model = self.xs = self.ys = None
        self.sim_key = None
        self.results, self.reports, self.runtime = {}, {}, {}
        self.stale = set()
        self.mc = None
        self.cursor = 0
        self.live = True
        self.runner = PoolRunner(self); self.mc_runner = PoolRunner(self)
        self.runner.partial.connect(self._on_partial)
        self.runner.done.connect(self._on_done)
        self.runner.failed.connect(self._on_failed)
        self.mc_runner.progressed.connect(self._on_mc_progress)
        self.mc_runner.done.connect(self._on_mc_done)
        self.mc_runner.failed.connect(self._on_failed)
        self._debounce = QtCore.QTimer(self); self._debounce.setSingleShot(True)
        self._debounce.setInterval(70); self._debounce.timeout.connect(self._launch)
        self._force = False
        self._build_controls()
        self._build_views()
        self.cb_sys.setCurrentIndex(0)
        self._system_changed()

    # ------------------------------------------------------------------ controls
    def _build_controls(self):
        c = self.controls
        f, v = card("Data", "simulate from a model, or load your own measurements "
                            "(columns t, measurements, optional x_<state> truth)")
        self.source = DataSource("series"); v.addWidget(self.source)
        c.addWidget(f)
        f, v = card("System")
        self.cb_sys = QtWidgets.QComboBox()
        self.catalog = {}
        self._fill_systems(list_ssm())
        v.addWidget(self.cb_sys)
        self.lbl_cat = muted(""); v.addWidget(self.lbl_cat)
        self.lbl_desc = muted(""); v.addWidget(self.lbl_desc)
        c.addWidget(f)

        f, v = card("Model parameters θ")
        self.lbl_theta = muted(""); self.lbl_theta.setWordWrap(True); v.addWidget(self.lbl_theta)
        self.params = ParamEditor(); v.addWidget(self.params)
        h = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Reset θ"); b.clicked.connect(lambda: self.params.reset())
        h.addWidget(b); h.addStretch(1); v.addLayout(h)
        fl = form()
        self.sp_T = spin(5, 20000, 100, 10); fl.addRow("steps T", self.sp_T)
        hs = QtWidgets.QHBoxLayout()
        self.sp_seed = spin(0, 10**6, 0)
        bn = QtWidgets.QPushButton("New"); bn.setToolTip("draw a new random data set")
        bn.clicked.connect(lambda: self.sp_seed.setValue(int(np.random.default_rng().integers(10**6))))
        hs.addWidget(self.sp_seed, 1); hs.addWidget(bn)
        fl.addRow("seed", hs)
        v.addLayout(fl)
        c.addWidget(f)

        f, v = card("Estimators")
        self.methods = MethodList(STATE_METHODS); self.methods.fit_height()
        v.addWidget(self.methods)
        self.cb_smooth = QtWidgets.QCheckBox("smoothers (RTS family / FFBSi)")
        self.cb_bmap = QtWidgets.QCheckBox("batch MAP trajectory (Gauss–Newton)")
        v.addWidget(self.cb_smooth); v.addWidget(self.cb_bmap)
        c.addWidget(f)

        f, v = card("Tuning")
        fl = form()
        self.sp_np = spin(10, 200000, 1000, 100); fl.addRow("particles", self.sp_np)
        self.cb_res = QtWidgets.QComboBox()
        self.cb_res.addItems(["systematic", "stratified", "residual", "multinomial"])
        fl.addRow("resampling", self.cb_res)
        self.sp_ens = spin(5, 5000, 100, 10); fl.addRow("ensemble", self.sp_ens)
        self.sp_infl = QtWidgets.QDoubleSpinBox(); self.sp_infl.setRange(1.0, 2.0)
        self.sp_infl.setSingleStep(0.01); self.sp_infl.setDecimals(3); self.sp_infl.setValue(1.0)
        fl.addRow("inflation", self.sp_infl)
        self.sp_alpha = QtWidgets.QDoubleSpinBox(); self.sp_alpha.setRange(1e-3, 2.0)
        self.sp_alpha.setDecimals(3); self.sp_alpha.setSingleStep(0.05); self.sp_alpha.setValue(1.0)
        fl.addRow("UKF α", self.sp_alpha)
        self.sp_h = spin(1, 200, 10); fl.addRow("MHE horizon", self.sp_h)
        self.cb_con = QtWidgets.QCheckBox("enforce state bounds"); self.cb_con.setChecked(True)
        fl.addRow("MHE", self.cb_con)
        v.addLayout(fl)
        c.addWidget(f)

        f, v = card("Monte Carlo", "independent data sets; consistency over runs")
        h = QtWidgets.QHBoxLayout()
        self.sp_runs = spin(2, 5000, 30, 10, " runs")
        self.b_mc = QtWidgets.QPushButton("Run Monte Carlo"); self.b_mc.clicked.connect(self.run_mc)
        h.addWidget(self.sp_runs); h.addWidget(self.b_mc, 1)
        v.addLayout(h)
        self.mc_bar = QtWidgets.QProgressBar(); self.mc_bar.setVisible(False); v.addWidget(self.mc_bar)
        c.addWidget(f)

        self.b_run = QtWidgets.QPushButton("Run all estimators"); self.b_run.setObjectName("primary")
        self.b_run.setToolTip("Ctrl+R — also runs slow methods that live mode skips")
        self.b_run.clicked.connect(self.run_all)
        c.addWidget(self.b_run)
        c.addStretch(1)

        self.cb_sys.currentIndexChanged.connect(lambda _: self._system_changed())
        self.source.changed.connect(self._source_changed)
        self.params.changed.connect(self.schedule)
        for w in (self.sp_T, self.sp_seed, self.sp_np, self.sp_ens, self.sp_h):
            w.valueChanged.connect(self.schedule)
        for w in (self.sp_infl, self.sp_alpha):
            w.valueChanged.connect(self.schedule)
        self.cb_res.currentIndexChanged.connect(self.schedule)
        for w in (self.cb_smooth, self.cb_bmap, self.cb_con):
            w.toggled.connect(self.schedule)
        self.methods.changed.connect(self.schedule)

    def _build_views(self):
        self.v_states = self.add_view("States", self.draw_states, reusable=True)
        self.v_track = self.add_view("Track", self.draw_track)
        self.v_err = self.add_view("Errors", self.draw_errors, reusable=True)
        self.v_cons = self.add_view("Consistency", self.draw_consistency, reusable=True)
        self.v_inn = self.add_view("Innovations", self.draw_innovations)
        self.v_part = self.add_view("Particles", self.draw_particles)
        self.v_meas = self.add_view("Measurements", self.draw_measurements, reusable=True)
        self.v_mc = self.add_view("Monte Carlo", self.draw_mc)
        self._cursor_lines, self._time_plots = {}, {}
        for v in (self.v_states, self.v_err, self.v_cons, self.v_part):
            v.glw.scene().sigMouseClicked.connect(self._scene_clicked)
        # state selector + cursor readout shared by time views
        self.sel_btn = QtWidgets.QToolButton(); self.sel_btn.setText("States ▾")
        self.sel_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.sel_menu = QtWidgets.QMenu(self); self.sel_btn.setMenu(self.sel_menu)
        self.readout = QtWidgets.QLabel(); self.readout.setObjectName("readout")
        self.readout.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.v_states.header.addWidget(self.sel_btn)
        hint = QtWidgets.QLabel("click / drag the cursor · ← → step"); hint.setObjectName("muted")
        self.v_states.header.addWidget(hint)
        self.v_states.header.addStretch(1)
        self.v_states.header.addWidget(self.readout)
        self.cb_comp = QtWidgets.QComboBox()
        self.cb_comp.currentIndexChanged.connect(lambda _: self.invalidate(self.v_part))
        self.cb_pf = QtWidgets.QComboBox()
        self.cb_pf.currentIndexChanged.connect(lambda _: self.invalidate(self.v_part))
        self.v_part.header.addWidget(QtWidgets.QLabel("state")); self.v_part.header.addWidget(self.cb_comp)
        self.v_part.header.addWidget(QtWidgets.QLabel("filter")); self.v_part.header.addWidget(self.cb_pf)
        self.v_part.header.addStretch(1)
        self.cb_nees_log = QtWidgets.QCheckBox("log scale"); self.cb_nees_log.setChecked(True)
        self.cb_nees_log.toggled.connect(lambda _: self.invalidate(self.v_cons))
        self.v_cons.header.addWidget(self.cb_nees_log)
        self.v_cons.header.addWidget(muted("shaded: pointwise 95 % χ² region"))
        self.v_cons.header.addStretch(1)
        self.set_table(ResultsTable(["method", "RMSE", "ANEES", "NEES in 95 %", "ANIS", "Ljung–Box p",
                                     "coverage 95 %", "log-lik", "time [ms]", "status"]))
        self.table.cellDoubleClicked.connect(self._solo)

    # ------------------------------------------------------------------ problem / data
    def _fill_systems(self, catalog):
        self.catalog = dict(catalog)
        self.cb_sys.blockSignals(True)
        self.cb_sys.clear()
        for name, p in self.catalog.items():
            self.cb_sys.addItem(name, name)
            self.cb_sys.setItemData(self.cb_sys.count() - 1, p.description, Qt.ToolTipRole)
        self.cb_sys.blockSignals(False)

    def data(self):
        """The loaded measurement file when "My data" is selected, else None."""
        return self.source.data if self.source.from_file() else None

    def _source_changed(self):
        d = self.data()
        self._fill_systems(data_problems(d) if d is not None else list_ssm())
        self.cb_sys.setCurrentIndex(0)
        sim = d is None
        for w in (self.sp_T, self.b_mc, self.sp_runs):
            w.setEnabled(sim)
        self.b_mc.setToolTip("" if sim else "Monte Carlo needs simulated data")
        self._system_changed()

    def _system_changed(self):
        p = self.catalog[self.cb_sys.currentData()]
        self.problem = p
        md = p.model()
        d = self.data()
        self.lbl_theta.setText("drag to re-simulate and re-estimate live" if d is None else
                               "the filters' model: drag to re-tune it on your data "
                               "(start values are estimated from the data)")
        self.lbl_cat.setText(f"{p.category} · nx = {md.nx}, ny = {md.ny}"
                             + ("" if md.gaussian_obs else " · non-Gaussian likelihood"))
        self.lbl_desc.setText(md.description or p.description)
        self.params.blockSignals(True)
        self.params.set_params(p.param_names, p.theta_true, p.lo, p.hi)
        self.params.blockSignals(False)
        self.sp_T.blockSignals(True); self.sp_T.setValue(p.T if d is None else d.T)
        self.sp_T.blockSignals(False)
        ok = [m for m in STATE_METHODS if applicable(m, md)]
        self.methods.blockSignals(True)
        self.methods.configure(ok, [m for m in p.recommended if m in ok])
        self.methods.blockSignals(False)
        self.sel_menu.clear()
        self.state_actions = []
        for i, n in enumerate(md.state_names):
            a = self.sel_menu.addAction(n); a.setCheckable(True); a.setChecked(i < MAX_PANELS)
            a.toggled.connect(lambda _: self.invalidate(self.v_states, self.v_err))
            self.state_actions.append(a)
        self.cb_comp.blockSignals(True); self.cb_comp.clear(); self.cb_comp.addItems(md.state_names)
        self.cb_comp.blockSignals(False)
        self.tabs.setTabEnabled(self.tabs.indexOf(self.v_track), p.xy is not None)
        self.results, self.reports, self.runtime, self.stale = {}, {}, {}, set()
        self.mc = None
        self.sim_key = None
        self.cursor = 0
        self.schedule()

    def theta(self):
        return np.clip(self.params.values(), self.problem.lo, self.problem.hi)

    def opts(self):
        return dict(n_particles=self.sp_np.value(), resampling=self.cb_res.currentText(),
                    n_ens=self.sp_ens.value(), inflation=self.sp_infl.value(),
                    alpha=self.sp_alpha.value(), horizon=self.sp_h.value(),
                    constrained=self.cb_con.isChecked())

    def selected_states(self):
        s = [i for i, a in enumerate(self.state_actions) if a.isChecked()]
        return s or [0]

    # ------------------------------------------------------------------ running
    def schedule(self, *_):
        self._debounce.start()

    def run_all(self):
        self._force = True
        self._launch()

    def cancel(self):
        self.runner.cancel(); self.mc_runner.cancel()
        self.say("cancelled")

    def set_live(self, on):
        self.live = on
        if on:
            self.schedule()

    def _launch(self):
        self._debounce.stop()
        force, self._force = self._force, False
        if not self.live and not force:
            self.status.emit("live updates off — press Run (Ctrl+R)")
            return
        p, th, T, seed = self.problem, self.theta(), self.sp_T.value(), self.sp_seed.value()
        data = self.data()
        key = (p.name, tuple(np.round(th, 12)), T, seed) if data is None else \
            ("file", self.source.path, id(data), p.name, tuple(np.round(th, 12)))
        need_sim = key != self.sim_key
        methods = self.methods.checked()
        opts, do_s, do_b = self.opts(), self.cb_smooth.isChecked(), self.cb_bmap.isChecked()
        okey = tuple(sorted(opts.items()))
        if not force:
            run = [m for m in methods if self.runtime.get(m, 0.0) <= LIVE_LIMIT]
        else:
            run = list(methods)
        skipped = [m for m in methods if m not in run]
        # drop results that no longer correspond to the data / selection / tuning
        invalid = need_sim or self._opts_key != okey
        keep = {}
        for k, r in self.results.items():
            base = k.split("+")[0]
            if invalid and base in skipped:
                continue
            if k == "batch-map" and do_b and not need_sim:
                keep[k] = r
            elif base in methods and (do_s or "+" not in k) and not need_sim:
                keep[k] = r
        self.results = keep
        self.stale = {m for m in skipped if m not in self.results}
        self._opts_key = okey
        bmap = do_b and (force or self.runtime.get("batch-map", 0.0) <= LIVE_LIMIT)
        if need_sim:          # simulation is cheap: do it here so the data appear at once
            model = p.model(th)
            if data is None:
                xs, ys = model.simulate(T, seed)
            else:             # measured data: truth only where the file has it (else NaN)
                ys = data.ys
                xs = full_truth(data, model)
                if xs is None:
                    xs = np.full((len(ys) + 1, model.nx), np.nan)
            self.sim_key, self.model, self.xs, self.ys = key, model, xs, ys
            self.results, self.reports = {}, {}
            self.cursor = min(self.cursor, len(ys))
            self._table_dirty = True
            self.invalidate(delay=200 if run else 10)
        bmap = bmap and self.model.gaussian_obs
        spec = p.name if data is None else problem_spec(p, data)
        tasks = [(state_task, (spec, th, self.ys, m, opts, seed, do_s)) for m in run]
        if bmap:
            tasks.append((state_task, (spec, th, self.ys, "batch-map", opts, seed, False)))
        self.runner.submit(tasks)
        self._t_launch = time.perf_counter()
        self.status.emit("running " + (", ".join(run) if run else "simulation") + " …"
                         if tasks else "ready")

    _opts_key = None

    def _on_partial(self, gen, key, obj):
        if not self.runner.is_current(gen):
            return
        if key == "_sim":
            self.sim_key, self.model, self.xs, self.ys = obj
            self.results = {}
            self.cursor = min(self.cursor, len(self.ys))
            self.reports = {}
            # new data: wait briefly for the first estimates instead of painting an empty
            # frame that would compete with the worker for the interpreter
            self._table_dirty = True
            self.invalidate(delay=180 if self.methods.checked() else 10)
            return
        res, dt = obj
        self.results[key] = res
        if "+" not in key:
            self.runtime[key] = dt
        self.stale.discard(key)
        self.reports.pop(key, None)
        self.invalidate(*[v for v in self.views if v is not self.v_mc and v is not self.v_meas])
        self._table_dirty = True

    def _on_done(self, gen, total):
        if not self.runner.is_current(gen):
            return
        ms = ", ".join(f"{k} {1e3 * self.runtime[k]:.0f}" for k in self.ordered() if k in self.runtime)
        wall = 1e3 * (time.perf_counter() - getattr(self, "_t_launch", time.perf_counter()))
        self.status.emit(f"done in {wall:.0f} ms wall-clock ({self.runner.mode} pool)"
                         + (f" — compute ms: {ms}" if ms else ""))

    def _on_failed(self, gen, msg):
        self.say("ERROR: " + msg)
        self.status.emit("error: " + msg.splitlines()[0])

    # ------------------------------------------------------------------ table
    def _report(self, k):
        if k not in self.reports:
            self.reports[k] = cs.report(self.xs, self.results[k])
        return self.reports[k]

    def _refresh_table(self):
        rows, keys, flags = [], [], []
        if self.xs is None:
            self.table.set_rows([]); return
        nx = self.xs.shape[1]
        lo1, hi1 = chi2.ppf([0.025, 0.975], nx)
        for k, r in self.ordered().items():
            rep = self._report(k)
            keys.append(k)
            if rep.get("diverged"):
                rows.append([k] + [np.nan] * 8 + ["diverged"]); flags.append({"row": "bad"}); continue
            if "anees" not in rep:            # no ground truth: judge by the innovations
                lb = float(np.min(rep["ljung_box_p"])) if "ljung_box_p" in rep else np.nan
                ll = getattr(r, "loglik", None)
                ms = 1e3 * (self.runtime.get(k, getattr(r, "runtime", np.nan)) if "+" not in k
                            else getattr(r, "runtime", np.nan))
                if "anis" in rep:
                    nlo, nhi = rep["anis_band"]
                    ok = nlo <= rep["anis"] <= nhi
                    st = "NIS consistent" if ok else ("NIS too large" if rep["anis"] > nhi
                                                      else "NIS too small")
                    tag = "good" if ok else ("bad" if rep["anis"] > nhi else "warn")
                else:
                    st, tag = "no truth", None
                rows.append([k, np.nan, np.nan, "", rep.get("anis", np.nan), lb, "",
                             float(ll) if ll is not None else np.nan, ms, st])
                flags.append({4: tag, 9: tag} if tag else {})
                continue
            alo, ahi = rep["anees_band"]
            lb = float(np.min(rep["ljung_box_p"])) if "ljung_box_p" in rep else np.nan
            ll = getattr(r, "loglik", None)
            ms = 1e3 * (self.runtime.get(k, getattr(r, "runtime", np.nan)) if "+" not in k
                        else getattr(r, "runtime", np.nan))
            ok = alo <= rep["anees"] <= ahi
            rows.append([k, rep["rmse"], rep["anees"], f"{100 * rep['nees_frac_in']:.0f} %",
                         rep.get("anis", np.nan), lb, f"{100 * rep['coverage95']:.0f} %",
                         float(ll) if ll is not None else np.nan, ms,
                         "consistent" if ok else ("overconfident" if rep["anees"] > ahi
                                                  else "conservative")])
            tag = "good" if ok else ("bad" if rep["anees"] > ahi else "warn")
            flags.append({2: tag, 9: tag})
        for m in sorted(self.stale):
            rows.append([m] + [np.nan] * 8 + [f"slow ({self.runtime.get(m, 0):.1f} s): press Run"])
            keys.append(m); flags.append({"row": "stale"})
        self.table.set_rows(rows, keys, flags)

    def _solo(self, row, col):
        """Double-click a table row: show only that method (double-click again: all)."""
        k = self.table.item(row, 0).text()
        self._solo_key = None if getattr(self, "_solo_key", None) == k else k
        self.invalidate()

    @staticmethod
    def _order(k):
        base = k.split("+")[0]
        order = list(STATE_METHODS) + ["batch-map"]
        return (order.index(base) if base in order else 99, k)

    def ordered(self):
        return {k: self.results[k] for k in sorted(self.results, key=self._order)}

    def shown(self):
        solo = getattr(self, "_solo_key", None)
        return {k: r for k, r in self.ordered().items()
                if (solo is None or k == solo) and np.all(np.isfinite(r.m))}

    # ------------------------------------------------------------------ cursor
    def t_axis(self):
        d = self.data()
        if d is not None and len(d.t) == len(self.xs) - 1:
            return np.r_[d.t[0] - d.dt, d.t]
        return np.arange(len(self.xs)) * self.model.dt

    def has_truth(self, i):
        return self.xs is not None and bool(np.all(np.isfinite(self.xs[1:, i])))

    def set_cursor_t(self, t):
        k = int(np.clip(np.argmin(np.abs(self.t_axis() - t)), 0, len(self.xs) - 1))
        if k != self.cursor:
            self.cursor = k
            self._update_cursor_items()

    def _update_cursor_items(self):
        tk = self.t_axis()[self.cursor]
        for lines in self._cursor_lines.values():
            for ln in lines:
                ln.blockSignals(True); ln.setValue(tk); ln.blockSignals(False)
        self._update_readout()
        for v in (self.v_track, self.v_part):
            v.dirty = True
        if self.current_view() in (self.v_track, self.v_part):
            self.refresh()

    def _update_readout(self):
        if self.xs is None:
            return
        k = self.cursor
        st = self.selected_states()[:2]
        parts = [f"k = {k}  t = {self.t_axis()[k]:.4g}"]
        for i in st:
            s = f"{self.model.state_names[i]}:" + (f" truth {self.xs[k, i]:.4g}"
                                                    if np.isfinite(self.xs[k, i]) else "")
            for n, r in list(self.shown().items())[:2]:
                sd = np.sqrt(max(r.P[k, i, i], 0))
                s += f" | {n} {r.m[k, i]:.4g}±{2 * sd:.2g}"
            parts.append(s)
        self.readout.setText("    ".join(parts))

    def keyPressEvent(self, e):
        if self.xs is not None and e.key() in (Qt.Key_Left, Qt.Key_Right):
            step = (10 if e.modifiers() & Qt.ShiftModifier else 1) * (1 if e.key() == Qt.Key_Right else -1)
            ta = self.t_axis()
            self.set_cursor_t(ta[int(np.clip(self.cursor + step, 0, len(ta) - 1))])
            return
        super().keyPressEvent(e)

    def _prune_cursor(self):
        pass

    def _cursor_line(self, plot, view=None):
        ln = pg.InfiniteLine(self.t_axis()[min(self.cursor, len(self.xs) - 1)], angle=90, movable=True,
                             pen=pg.mkPen(theme.current["accent"], width=1.5),
                             hoverPen=pg.mkPen(theme.current["accent"], width=3))
        ln.sigPositionChanged.connect(lambda l: self.set_cursor_t(l.value()))
        plot.addItem(ln, ignoreBounds=True)
        view = view or self.current_view()
        self._cursor_lines.setdefault(view, []).append(ln)
        self._time_plots.setdefault(view, []).append(plot)

    def _scene_clicked(self, ev):
        if ev.button() != Qt.LeftButton or self.xs is None:
            return
        for plot in self._time_plots.get(self.current_view(), []):
            try:
                if plot.scene() is not None and plot.vb.sceneBoundingRect().contains(ev.scenePos()):
                    self.set_cursor_t(plot.vb.mapSceneToView(ev.scenePos()).x())
                    return
            except RuntimeError:          # plot already deleted
                continue

    # ------------------------------------------------------------------ drawing
    def _need_data(self):
        if self.xs is None:
            raise Empty("simulating …")

    def _stack(self, view, rows, ylabels, legend=True):
        """Stacked, x-linked time plots; reused while the layout key is unchanged."""
        self._cursor_lines[view], self._time_plots[view] = [], []
        key = (self.problem.name, tuple(rows), theme.current["name"])
        plots = view.begin(key)
        if plots is None:
            plots = []
            for row, lab in enumerate(ylabels):
                last = row == len(ylabels) - 1
                p = new_plot(view.glw, row, ylabel=lab, legend=legend and row == 0,
                             xlabel="time" if last else None, xvalues=last)
                if plots:
                    p.setXLink(plots[0])
                plots.append(p)
            view.keep(plots)
        return plots

    def draw_states(self, glw):
        self._need_data()
        t = self.t_axis()
        res = self.shown()
        sel = self.selected_states()
        plots = self._stack(self.v_states, sel, [self.model.state_names[i] for i in sel])
        for p, i in zip(plots, sel):
            for n, r in res.items():
                sd = np.sqrt(np.maximum(r.P[:, i, i], 0))
                band(p, t, r.m[:, i] - 2 * sd, r.m[:, i] + 2 * sd, n, 30)
            if self.has_truth(i):
                p.plot(t, self.xs[:, i], pen=truth_pen(), name="truth", connect="finite")
            for n, r in res.items():
                p.plot(t, r.m[:, i], pen=pen(n, 1.6, method_style(n)), name=n)
            self._cursor_line(p, self.v_states)
        self._update_readout()

    def draw_errors(self, glw):
        self._need_data()
        if not self.results:
            raise Empty("select estimators to see errors ± 2σ")
        t = self.t_axis()
        sel = [i for i in self.selected_states() if self.has_truth(i)]
        if not sel:
            raise Empty("errors need ground truth: add x_<state> columns to your file "
                        "(for the states selected under States ▾)")
        plots = self._stack(self.v_err, sel, [f"error {self.model.state_names[i]}" for i in sel])
        for p, i in zip(plots, sel):
            for n, r in self.shown().items():
                e = self.xs[:, i] - r.m[:, i]
                sd = np.sqrt(np.maximum(r.P[:, i, i], 0))
                band(p, t, -2 * sd, 2 * sd, n, 28)
                p.plot(t, e, pen=pen(n, 1.2, method_style(n)), name=n, connect="finite")
            p.addLine(y=0, pen=pg.mkPen(theme.current["muted"], width=1))
            self._cursor_line(p, self.v_err)

    def draw_track(self, glw):
        self._need_data()
        if self.problem.xy is None:
            raise Empty("this system has no planar track")
        p = new_plot(glw, 0, legend=True, xlabel=self.model.state_names[self.problem.xy[0]],
                     ylabel=self.model.state_names[self.problem.xy[1]])
        p.setAspectLocked(True)
        i, j = self.problem.xy
        k = self.cursor
        if self.problem.meas_xy is not None:
            m = self.problem.meas_xy(self.ys)
            p.plot(m[:, 0], m[:, 1], pen=None, symbol="o", symbolSize=3,
                   symbolBrush=pg.mkBrush(theme.current["band"]), symbolPen=None, name="measurements")
        for name, pts in self.problem.landmarks.items():
            pts = np.asarray(pts)
            if len(pts) > 20:
                p.plot(pts[:, 0], pts[:, 1], pen=pg.mkPen(theme.current["muted"], width=1.4,
                                                          style=Qt.DotLine), name=name)
                kk = min(k, len(pts) - 1)
                p.plot([pts[kk, 0]], [pts[kk, 1]], pen=None, symbol="t1", symbolSize=11,
                       symbolBrush=pg.mkBrush(theme.current["muted"]))
            else:
                p.plot(pts[:, 0], pts[:, 1], pen=None, symbol="t", symbolSize=12,
                       symbolBrush=pg.mkBrush(theme.current["truth"]), name=name)
        if self.has_truth(i) and self.has_truth(j):
            p.plot(self.xs[1:, i], self.xs[1:, j], pen=truth_pen(), name="truth")
            if k > 0:
                p.plot([self.xs[k, i]], [self.xs[k, j]], pen=None, symbol="o", symbolSize=9,
                       symbolBrush=pg.mkBrush(theme.current["truth"]))
        for n, r in self.shown().items():
            p.plot(r.m[:, i], r.m[:, j], pen=pen(n, 1.5, method_style(n)), name=n)
            parts = getattr(r, "particles", None)
            if parts is not None:
                p.plot(parts[k, :, i], parts[k, :, j], pen=None, symbol="o", symbolSize=2,
                       symbolPen=None, symbolBrush=pg.mkBrush(theme.qcolor(n, 90)))
            ex, ey = ellipse(r.m[k, [i, j]], r.P[k][np.ix_([i, j], [i, j])])
            p.plot(ex, ey, pen=pen(n, 2.0))
            p.plot([r.m[k, i]], [r.m[k, j]], pen=None, symbol="x", symbolSize=9,
                   symbolPen=pen(n, 2))
        p.setTitle(f"k = {k}: estimates with 2σ ellipses (move the cursor in States or with ← →)",
                   size="8pt", color=theme.current["muted"])

    def draw_consistency(self, glw):
        self._need_data()
        if not self.results:
            raise Empty("select estimators to see NEES / NIS")
        t = self.t_axis()[1:]
        nx, ny = self.model.nx, self.model.ny
        logy = self.cb_nees_log.isChecked()
        self._cursor_lines[self.v_cons], self._time_plots[self.v_cons] = [], []
        plots = self.v_cons.begin((self.problem.name, logy, theme.current["name"]))
        if plots is None:
            p1 = new_plot(glw, 0, ylabel="NEES", legend=True, xvalues=False)
            p2 = new_plot(glw, 1, ylabel="NIS", xlabel="time")
            p2.setXLink(p1)
            self.v_cons.keep([p1, p2])
        else:
            p1, p2 = plots
        p2.setTitle(None)
        cols = cs.truth_columns(self.xs)
        p1.setLogMode(y=logy)
        if cols:
            nx = len(cols)
            X = self.xs[:, cols]
            nees = {n: cs.nees(X, r.m[:, cols], r.P[:, cols][:, :, cols])
                    for n, r in self.shown().items()}
            lo, hi = chi2.ppf([0.025, 0.975], nx)
            hband(p1, np.log10(lo) if logy else lo, np.log10(hi) if logy else hi)
            for n, e in nees.items():
                p1.plot(t, np.maximum(e, 1e-6), pen=pen(n, 1.1, method_style(n)), name=n)
            p1.addLine(y=np.log10(nx) if logy else nx,
                       pen=pg.mkPen(theme.current["muted"], style=Qt.DashLine))
            self._robust_range(p1, list(nees.values()), nx, logy)
            p1.setTitle(None if nx == self.model.nx else
                        f"NEES over the {nx} states with ground truth", size="8pt")
        else:
            p1.setTitle("NEES needs ground truth (x_<state> columns in your file); "
                        "NIS below needs none", size="8pt")
            for n in self.shown():          # keep the legend
                p1.plot([], [], pen=pen(n, 1.1, method_style(n)), name=n)
        lo, hi = chi2.ppf([0.025, 0.975], ny)
        hband(p2, np.log10(lo) if logy else lo, np.log10(hi) if logy else hi)
        p2.setLogMode(y=logy)
        has = False
        for n, r in self.shown().items():
            v, S = getattr(r, "v", None), getattr(r, "S", None)
            if v is None or S is None or not np.all(np.isfinite(S)):
                continue
            has = True
            p2.plot(t, np.maximum(cs.nis(v, S), 1e-6), pen=pen(n, 1.1, method_style(n)))
        if not has:
            p2.setTitle("no innovations (smoothers / non-Gaussian likelihood)", size="8pt")
        else:
            self._robust_range(p2, [cs.nis(r.v, r.S) for r in self.shown().values()
                                    if getattr(r, "v", None) is not None and getattr(r, "S", None)
                                    is not None and np.all(np.isfinite(r.S))], ny, logy)
        self._cursor_line(p1, self.v_cons); self._cursor_line(p2, self.v_cons)

    @staticmethod
    def _robust_range(p, series, dof, logy):
        """Keep the χ² region in view; a degenerate method (NEES ~ 1e9) must not squash the rest."""
        lo, hi = chi2.ppf([0.001, 0.999], dof)
        if series:
            allv = np.concatenate([np.asarray(s)[np.isfinite(s)] for s in series] + [np.array([hi])])
            top = min(np.percentile(allv, 99), 1e3 * dof)
            bot = max(np.percentile(allv, 1), 1e-3 * dof)
            lo, hi = min(lo, bot), max(hi, top)
        if logy:
            p.setYRange(np.log10(lo), np.log10(hi), padding=0.05)
        else:
            p.setYRange(0, hi, padding=0.05)

    def draw_innovations(self, glw):
        self._need_data()
        res = {n: r for n, r in self.shown().items()
               if getattr(r, "v", None) is not None and getattr(r, "S", None) is not None
               and np.all(np.isfinite(r.S))}
        if not res:
            raise Empty("innovations come from Gaussian / ensemble filters")
        t = self.t_axis()[1:]
        p1 = new_plot(glw, 0, ylabel="whitened innov. (1st)", legend=True, xlabel="time")
        hband(p1, -1.96, 1.96, 30)
        p2 = new_plot(glw, 1, ylabel="autocorrelation", xlabel="lag")
        w = 0.8 / len(res)
        for j, (n, r) in enumerate(res.items()):
            z = cs.whiten(r.v, r.S)
            p1.plot(t, z[:, 0], pen=pen(n, 1.0), name=n)
            ac, bound = cs.autocorr(z, min(20, len(z) // 4))
            lags = np.arange(1, len(ac) + 1) + (j - (len(res) - 1) / 2) * w
            p2.addItem(pg.BarGraphItem(x=lags, height=ac, width=w * 0.95,
                                       brush=pg.mkBrush(theme.qcolor(n, 200)), pen=pg.mkPen(None)))
        hband(p2, -bound, bound, 35)
        p2.addLine(y=0, pen=pg.mkPen(theme.current["muted"]))

    def draw_particles(self, glw):
        self._need_data()
        pfs = {n: r for n, r in self.results.items()
               if getattr(r, "particles", None) is not None and not getattr(r, "diverged", False)}
        names = list(pfs)
        cur = self.cb_pf.currentText()
        self.cb_pf.blockSignals(True); self.cb_pf.clear(); self.cb_pf.addItems(names)
        if cur in names:
            self.cb_pf.setCurrentText(cur)
        self.cb_pf.blockSignals(False)
        if not pfs:
            raise Empty("run a particle filter (pf, rpf or pf-ekf) to see particle densities")
        name = self.cb_pf.currentText() or names[0]
        pf = pfs[name]
        i = max(self.cb_comp.currentIndex(), 0)
        t = self.t_axis()
        X = pf.particles[:, :, i]
        W = np.exp(pf.logw) if pf.logw is not None else np.full(X.shape, 1.0 / X.shape[1])
        lo, hi = np.nanpercentile(X, [0.5, 99.5])
        if self.has_truth(i):
            lo = min(lo, np.nanmin(self.xs[:, i])); hi = max(hi, np.nanmax(self.xs[:, i]))
        pad = 0.05 * (hi - lo + 1e-12); lo -= pad; hi += pad
        nb = 120
        idx = np.clip(((X - lo) / (hi - lo) * nb).astype(int), 0, nb - 1)
        H = np.zeros((len(t), nb))
        np.add.at(H, (np.repeat(np.arange(len(t)), X.shape[1]), idx.ravel()), W.ravel())
        H = np.sqrt(H / np.maximum(H.max(1, keepdims=True), 1e-300))
        p = new_plot(glw, 0, 0, ylabel=self.model.state_names[i], xlabel="time", legend=True)
        img = pg.ImageItem(H.T)
        img.setLookupTable(theme.heat_lut())
        dt = (t[-1] - t[0]) / max(len(t) - 1, 1)
        img.setRect(QtCore.QRectF(t[0] - 0.5 * dt, lo, t[-1] - t[0] + dt, hi - lo))
        p.addItem(img)
        sm = self.results.get(name + "+ffbsi")
        if sm is not None and sm.samples is not None:
            for s in sm.samples[:15]:
                p.plot(t, s[:, i], pen=pg.mkPen(theme.current["band"], width=0.7))
        if self.has_truth(i):
            p.plot(t, self.xs[:, i], pen=truth_pen(1.6), name="truth", connect="finite")
        p.plot(t, pf.m[:, i], pen=pen(name, 1.4), name=f"{name} mean")
        self._cursor_lines[self.v_part], self._time_plots[self.v_part] = [], []
        self._cursor_line(p, self.v_part)
        # density slice at the cursor vs Gaussian estimates
        k = self.cursor
        q = new_plot(glw, 0, 1, xlabel="density at cursor", title=f"k = {k}")
        q.getAxis("left").setWidth(40)
        q.setYLink(p)
        glw.ci.layout.setColumnStretchFactor(0, 5); glw.ci.layout.setColumnStretchFactor(1, 1)
        edges = np.linspace(lo, hi, 61)
        h, _ = np.histogram(X[k], bins=edges, weights=W[k], density=True)
        yc = 0.5 * (edges[1:] + edges[:-1])
        c1 = q.plot(h, yc, pen=pen(name, 1.5))
        c0 = pg.PlotDataItem(np.zeros_like(h), yc, pen=None)
        q.addItem(pg.FillBetweenItem(c1, c0, brush=pg.mkBrush(theme.qcolor(name, 60))))
        yy = np.linspace(lo, hi, 300)
        for n, r in self.shown().items():
            if n == name or getattr(r, "particles", None) is not None:
                continue
            sd = np.sqrt(max(r.P[k, i, i], 1e-300))
            q.plot(np.exp(-0.5 * ((yy - r.m[k, i]) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), yy,
                   pen=pen(n, 1.2, method_style(n)))
        if np.isfinite(self.xs[k, i]):
            q.addLine(y=self.xs[k, i], pen=truth_pen(1.2))

    def draw_measurements(self, glw):
        self._need_data()
        t = self.t_axis()[1:]
        ny = self.model.ny
        show = list(range(min(ny, MAX_PANELS)))
        yt = self.model.h(self.xs[1:], np.arange(1, len(self.xs))[:, None]) \
            if (self.model.gaussian_obs or hasattr(self.model, "h")) and \
            len(cs.truth_columns(self.xs)) == self.model.nx else None
        key = (self.problem.name, tuple(show), theme.current["name"])
        plots = self.v_meas.begin(key)
        if plots is None:
            plots = []
            for row, j in enumerate(show):
                last = row == len(show) - 1
                p = new_plot(glw, row, ylabel=self.model.obs_names[j], legend=row == 0,
                             xlabel="time" if last else None, xvalues=last)
                if plots:
                    p.setXLink(plots[0])
                plots.append(p)
            self.v_meas.keep(plots)
        for p, j in zip(plots, show):
            p.plot(t, self.ys[:, j], pen=None, symbol="o", symbolSize=3, symbolPen=None,
                   symbolBrush=pg.mkBrush(theme.current["band"]), name="measurement")
            if yt is not None:
                p.plot(t, yt[:, j], pen=truth_pen(1.3), name="h(truth)")
        if ny > len(show):
            plots[-1].setTitle(f"showing {len(show)} of {ny} measurement components", size="8pt")

    # ------------------------------------------------------------------ Monte Carlo
    def run_mc(self):
        if self.mc_runner.running():
            self.mc_runner.cancel(); self.b_mc.setText("Run Monte Carlo"); self.mc_bar.setVisible(False)
            self.status.emit("Monte Carlo cancelled")
            return
        methods, opts, p, th = self.methods.checked(), self.opts(), self.problem, self.theta()
        R, T, seed, sm = self.sp_runs.value(), self.sp_T.value(), self.sp_seed.value(), \
            self.cb_smooth.isChecked()
        if not methods:
            self.status.emit("select at least one estimator"); return

        n_chunks = max(1, min(R, 3 * self.mc_runner.workers))
        chunks = [c.tolist() for c in np.array_split(np.arange(R), n_chunks) if len(c)]
        tasks = [(mc_task, (p.name, methods, R, c, T, seed, th, sm, opts)) for c in chunks]
        self.mc_bar.setRange(0, len(tasks)); self.mc_bar.setValue(0); self.mc_bar.setVisible(True)
        self.b_mc.setText("Cancel")
        self._t_mc = time.perf_counter()
        self.mc_runner.submit(tasks, combine=merge_mc)
        self.status.emit(f"Monte Carlo: {R} runs of {', '.join(methods)} on "
                         f"{self.mc_runner.workers} worker processes …")

    def _on_mc_progress(self, gen, i, n):
        self.mc_bar.setRange(0, n); self.mc_bar.setValue(i)

    def _on_mc_done(self, gen, mc):
        if mc is None:
            self.mc_bar.setVisible(False); self.b_mc.setText("Run Monte Carlo")
            return
        self.mc = mc
        self.mc_bar.setVisible(False); self.b_mc.setText("Run Monte Carlo")
        S = mc["_summary"]
        self.say("Monte Carlo summary (median-over-time ANEES):")
        for m, s in S.items():
            if s.get("n_ok"):
                self.say(f"  {m:12s} RMSE {s['rmse']:.4g}  ANEES {s['anees_median']:.3g}  "
                         f"in band {100 * s['anees_frac_in']:.0f} %  diverged {s['n_diverged']}  "
                         f"{s['time_ms']:.1f} ms")
            else:
                self.say(f"  {m:12s} all runs diverged")
        self.status.emit(f"Monte Carlo done: {mc['_meta']['n_runs']} runs in "
                         f"{time.perf_counter() - self._t_mc:.1f} s")
        self.tabs.setCurrentWidget(self.v_mc)
        self.invalidate(self.v_mc)

    def draw_mc(self, glw):
        if self.mc is None:
            raise Empty("press Run Monte Carlo")
        mc, S = self.mc, self.mc["_summary"]
        R, nx, ny = mc["_meta"]["n_runs"], mc["_meta"]["nx"], mc["_meta"]["ny"]
        names = [m for m in S if S[m].get("n_ok")]
        p1 = new_plot(glw, 0, 0, ylabel=f"ANEES ({R} runs)", legend=True, xlabel="step", )
        p1.setLogMode(y=True)
        lo, hi = chi2.ppf([0.025, 0.975], nx * R) / R
        hband(p1, np.log10(lo), np.log10(hi))
        for m in names:
            d = mc[m]
            a = np.nanmean(d["nees"][~d["diverged"]], axis=0)
            p1.plot(np.arange(1, len(a) + 1), np.maximum(a, 1e-6), pen=pen(m, 1.3, method_style(m)),
                    name=m)
        sub = glw.addLayout(row=2, col=0)
        x = np.arange(len(names))
        for col, (key, lab, logy) in enumerate([("rmse", "RMSE", False),
                                                ("anees_median", "median ANEES / nx", True),
                                                ("time_ms", "time per run [ms]", True)]):
            q = sub.addPlot(row=0, col=col)
            q.showGrid(y=True, alpha=0.18)
            vals = np.array([S[m][key] / (nx if key == "anees_median" else 1) for m in names], float)
            base = 0.0
            if logy:
                vals = np.log10(np.maximum(vals, 1e-9))
                base = min(np.floor(np.nanmin(vals)), 0.0) if len(vals) else 0.0
                q.getAxis("left").setLogMode(True)
            q.addItem(pg.BarGraphItem(x=x, y0=base, y1=vals, width=0.7,
                                      brushes=[pg.mkBrush(theme.qcolor(m, 220)) for m in names],
                                      pen=pg.mkPen(None)))
            if key == "anees_median":
                q.addLine(y=0.0, pen=pg.mkPen(theme.current["muted"], style=Qt.DashLine))
            q.setTitle(lab, size="8pt")
            bar_ticks(q, names)
        glw.ci.layout.setRowStretchFactor(1, 3); glw.ci.layout.setRowStretchFactor(2, 2)
