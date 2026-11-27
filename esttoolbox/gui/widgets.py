"""Shared GUI building blocks: background runner, parameter sliders, cards, tables, plots."""
from __future__ import annotations

import csv
import math
import traceback

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from . import theme


# =========================================================================== background jobs
class Cancelled(Exception):
    pass


class JobContext:
    """Handed to a job function; emitting from a cancelled job raises Cancelled."""

    def __init__(self, runner, gen):
        self.gen, self.cancelled, self._r = gen, False, runner

    def check(self):
        if self.cancelled:
            raise Cancelled

    def emit(self, key, obj):
        self.check()
        self._r.partial.emit(self.gen, key, obj)

    def progress(self, i, n):
        self.check()
        self._r.progressed.emit(self.gen, int(i), int(n))

    def log(self, msg):
        self._r.logged.emit(self.gen, msg)


class _JobThread(QtCore.QThread):
    def __init__(self, fn, ctx, runner):
        super().__init__()
        self.fn, self.ctx, self.runner = fn, ctx, runner

    def run(self):
        try:
            out = self.fn(self.ctx)
            if not self.ctx.cancelled:
                self.runner.done.emit(self.ctx.gen, out)
        except Cancelled:
            pass
        except Exception as e:  # surfaced in the log
            if not self.ctx.cancelled:
                self.runner.failed.emit(self.ctx.gen, f"{type(e).__name__}: {e}\n"
                                                      f"{traceback.format_exc()}")


class Runner(QtCore.QObject):
    """Runs one job at a time in a QThread. Submitting a new job cancels the previous one
    (cooperatively, at its next emit/progress/check) and results carry a generation number,
    so stale results are never applied."""
    partial = QtCore.pyqtSignal(int, str, object)
    done = QtCore.pyqtSignal(int, object)
    failed = QtCore.pyqtSignal(int, str)
    progressed = QtCore.pyqtSignal(int, int, int)
    logged = QtCore.pyqtSignal(int, str)
    busy = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gen = 0
        self.ctx = None
        self.threads = set()

    def submit(self, fn):
        if self.ctx is not None:
            self.ctx.cancelled = True
        self.gen += 1
        self.ctx = JobContext(self, self.gen)
        t = _JobThread(fn, self.ctx, self)
        self.threads.add(t)
        t.finished.connect(self._finished)
        t.start()
        self.busy.emit(True)
        return self.gen

    def cancel(self):
        if self.ctx is not None:
            self.ctx.cancelled = True

    def _finished(self):
        t = self.sender()
        self.threads.discard(t)
        if t is not None:
            t.deleteLater()
        if not self.running():
            self.busy.emit(False)

    def running(self):
        return any(t.isRunning() for t in self.threads)

    def is_current(self, gen):
        return gen == self.gen

    def wait_idle(self, ms=600_000):
        app = QtWidgets.QApplication.instance()
        timer = QtCore.QElapsedTimer(); timer.start()
        while (self.running() or self.threads) and timer.elapsed() < ms:
            app.processEvents()
            QtCore.QThread.msleep(5)
        app.processEvents()


# =========================================================================== small widgets
def card(title=None, subtitle=None):
    """A rounded panel with an optional title; returns (frame, inner QVBoxLayout)."""
    f = QtWidgets.QFrame(); f.setObjectName("card")
    v = QtWidgets.QVBoxLayout(f); v.setContentsMargins(12, 10, 12, 12); v.setSpacing(6)
    if title:
        lab = QtWidgets.QLabel(title); lab.setObjectName("cardTitle"); v.addWidget(lab)
    if subtitle:
        s = QtWidgets.QLabel(subtitle); s.setObjectName("muted"); s.setWordWrap(True); v.addWidget(s)
    return f, v


def form():
    fl = QtWidgets.QFormLayout()
    fl.setLabelAlignment(Qt.AlignLeft)
    fl.setHorizontalSpacing(10); fl.setVerticalSpacing(5)
    fl.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
    return fl


def spin(lo, hi, val, step=1, suffix=""):
    s = QtWidgets.QSpinBox(); s.setRange(lo, hi); s.setValue(val); s.setSingleStep(step)
    s.setSuffix(suffix); s.setKeyboardTracking(False); s.setAccelerated(True)
    return s


def muted(text):
    lab = QtWidgets.QLabel(text); lab.setObjectName("muted"); lab.setWordWrap(True)
    return lab


class SciSpin(QtWidgets.QDoubleSpinBox):
    """Double spin box that shows 5 significant digits and accepts 1e-3 notation;
    the step is 10 % of the value (multiplicative), which suits scale parameters."""

    def __init__(self, lo, hi, val, parent=None):
        super().__init__(parent)
        self.setDecimals(12); self.setRange(lo, hi); self.setValue(val)
        self.setKeyboardTracking(False); self.setAccelerated(True)

    def textFromValue(self, v):
        return f"{v:.4g}"

    def valueFromText(self, text):
        try:
            return float(text)
        except ValueError:
            return self.value()

    def validate(self, text, pos):
        try:
            float(text)
            return QtGui.QValidator.Acceptable, text, pos
        except ValueError:
            return QtGui.QValidator.Intermediate, text, pos

    def stepBy(self, n):
        v = self.value()
        step = 0.1 * abs(v) if v != 0 else 0.01 * (self.maximum() - self.minimum())
        self.setValue(v + n * step)


class ParamEditor(QtWidgets.QWidget):
    """One row per parameter: name, slider (log scale for positive ranges spanning > 50×),
    and an editable value. Emits `changed` continuously while dragging."""
    changed = QtCore.pyqtSignal()
    N = 1000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0); self.grid.setHorizontalSpacing(8)
        self.grid.setVerticalSpacing(4)
        self.rows = []

    def set_params(self, names, values, lo, hi, defaults=None):
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w is not None:
                w.hide(); w.setParent(None); w.deleteLater()
        self.rows = []
        self.defaults = np.array(values if defaults is None else defaults, float)
        for i, (n, v, a, b) in enumerate(zip(names, values, lo, hi)):
            logs = a > 0 and b / a > 50
            lab = QtWidgets.QLabel(n)
            lab.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
            sl = QtWidgets.QSlider(Qt.Horizontal); sl.setRange(0, self.N); sl.setMinimumWidth(70)
            sp = SciSpin(float(a), float(b), float(v))
            sp.setFixedWidth(96)
            row = dict(lo=float(a), hi=float(b), log=logs, slider=sl, spin=sp)
            self.rows.append(row)
            self._sync_slider(row)
            sl.valueChanged.connect(lambda pos, r=row: self._from_slider(r, pos))
            sp.valueChanged.connect(lambda val, r=row: self._from_spin(r))
            self.grid.addWidget(lab, i, 0); self.grid.addWidget(sl, i, 1); self.grid.addWidget(sp, i, 2)
        self.grid.setColumnStretch(1, 1)

    def _to_pos(self, r, v):
        if r["log"]:
            u = (math.log(v) - math.log(r["lo"])) / (math.log(r["hi"]) - math.log(r["lo"]))
        else:
            u = (v - r["lo"]) / (r["hi"] - r["lo"])
        return int(round(min(max(u, 0.0), 1.0) * self.N))

    def _from_pos(self, r, pos):
        u = pos / self.N
        if r["log"]:
            return math.exp(math.log(r["lo"]) + u * (math.log(r["hi"]) - math.log(r["lo"])))
        return r["lo"] + u * (r["hi"] - r["lo"])

    def _sync_slider(self, r):
        r["slider"].blockSignals(True)
        r["slider"].setValue(self._to_pos(r, r["spin"].value()))
        r["slider"].blockSignals(False)

    def _from_slider(self, r, pos):
        v = float(f"{self._from_pos(r, pos):.4g}")
        r["spin"].blockSignals(True); r["spin"].setValue(v); r["spin"].blockSignals(False)
        self.changed.emit()

    def _from_spin(self, r):
        self._sync_slider(r)
        self.changed.emit()

    def values(self):
        return np.array([r["spin"].value() for r in self.rows])

    def set_values(self, vals, emit=True):
        for r, v in zip(self.rows, vals):
            r["spin"].blockSignals(True); r["spin"].setValue(float(v)); r["spin"].blockSignals(False)
            self._sync_slider(r)
        if emit:
            self.changed.emit()

    def reset(self):
        self.set_values(self.defaults)


class MethodList(QtWidgets.QListWidget):
    """Checkable list with colour swatches; disabled rows for inapplicable methods."""
    changed = QtCore.pyqtSignal()

    def __init__(self, methods, parent=None):
        super().__init__(parent)
        for k, desc in methods.items():
            it = QtWidgets.QListWidgetItem(theme.swatch(k), k)
            it.setToolTip(desc)
            it.setData(Qt.UserRole, k)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self.addItem(it)
        self.itemChanged.connect(lambda _: self.changed.emit())
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.itemClicked.connect(self._toggle)

    def _toggle(self, it):
        pass

    def configure(self, enabled, checked):
        self.blockSignals(True)
        for i in range(self.count()):
            it = self.item(i)
            k = it.data(Qt.UserRole)
            ok = k in enabled
            it.setFlags((it.flags() | Qt.ItemIsEnabled) if ok else (it.flags() & ~Qt.ItemIsEnabled))
            it.setCheckState(Qt.Checked if ok and k in checked else Qt.Unchecked)
        self.blockSignals(False)
        self.changed.emit()

    def checked(self):
        return [self.item(i).data(Qt.UserRole) for i in range(self.count())
                if self.item(i).checkState() == Qt.Checked and self.item(i).flags() & Qt.ItemIsEnabled]

    def set_checked(self, names):
        self.blockSignals(True)
        for i in range(self.count()):
            it = self.item(i)
            if it.flags() & Qt.ItemIsEnabled:
                it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in names else Qt.Unchecked)
        self.blockSignals(False)
        self.changed.emit()

    def fit_height(self):
        h = sum(self.sizeHintForRow(i) for i in range(self.count())) + 2 * self.frameWidth() + 4
        self.setFixedHeight(min(h, 340))


class ResultsTable(QtWidgets.QTableWidget):
    def __init__(self, headers, parent=None):
        super().__init__(0, len(headers), parent)
        self.headers = headers
        self.setHorizontalHeaderLabels(headers)
        hh = self.horizontalHeader()
        hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        hh.setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(22)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setShowGrid(False)
        self.setAlternatingRowColors(False)

    def set_rows(self, rows, keys=None, flags=None):
        """rows: list of lists; keys: method name per row for the colour swatch;
        flags: per-row dict(col -> 'good'|'bad'|'stale')."""
        self.setSortingEnabled(False)
        self.setRowCount(len(rows))
        muted = QtGui.QColor(theme.current["muted"])
        for i, row in enumerate(rows):
            fl = (flags or [{}] * len(rows))[i] or {}
            for j, v in enumerate(row):
                if isinstance(v, (float, np.floating)):
                    txt = "—" if not np.isfinite(v) else f"{v:.4g}"
                    it = QtWidgets.QTableWidgetItem(txt)
                    it.setData(Qt.UserRole, float(v))
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                else:
                    it = QtWidgets.QTableWidgetItem(str(v))
                if j == 0 and keys is not None and keys[i]:
                    it.setIcon(theme.swatch(keys[i]))
                tag = fl.get(j) or fl.get("row")
                if tag == "good":
                    it.setForeground(QtGui.QColor("#16a34a"))
                elif tag == "bad":
                    it.setForeground(QtGui.QColor("#dc2626"))
                elif tag == "warn":
                    it.setForeground(QtGui.QColor("#d97706"))
                elif tag == "stale":
                    it.setForeground(muted)
                self.setItem(i, j, it)

    def to_csv(self, path):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(self.headers)
            for i in range(self.rowCount()):
                w.writerow([self.item(i, j).text() if self.item(i, j) else ""
                            for j in range(self.columnCount())])


# =========================================================================== plotting helpers
def pen(name_or_color, width=1.6, style=Qt.SolidLine, alpha=255):
    c = QtGui.QColor(theme.color(name_or_color) if not str(name_or_color).startswith("#")
                     else name_or_color)
    c.setAlpha(alpha)
    p = pg.mkPen(c, width=width)
    p.setStyle(style)
    p.setCosmetic(True)
    return p


def truth_pen(width=2.0):
    return pg.mkPen(theme.current["truth"], width=width)


def method_style(name):
    return Qt.DashLine if "+" in name else Qt.SolidLine


def new_plot(glw, row, col=0, title=None, xlabel=None, ylabel=None, legend=False, xvalues=True):
    """PlotItem at (row + 1, col): row 0 of every layout is reserved for a legend strip,
    so legends never cover data. legend=True attaches the strip to this plot."""
    p = glw.addPlot(row=row + 1, col=col)
    p.showGrid(x=True, y=True, alpha=0.18)
    for ax in ("left", "bottom"):
        p.getAxis(ax).enableAutoSIPrefix(False)
    if title:
        p.setTitle(title, size="9pt", color=theme.current["plot_fg"])
    if xlabel:
        p.setLabel("bottom", xlabel)
    if ylabel:
        p.setLabel("left", ylabel)
    if not xvalues:
        p.getAxis("bottom").setStyle(showValues=False)
        p.getAxis("bottom").setHeight(8)
    p.getAxis("left").setWidth(56)
    glw.ci.layout.setRowStretchFactor(row + 1, 1)
    if legend:
        lg = pg.LegendItem(colCount=12, labelTextSize="8pt", horSpacing=14, verSpacing=-6)
        glw.addItem(lg, row=0, col=0, colspan=2)
        lg.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        lg.setMaximumHeight(46)
        glw.ci.layout.setRowStretchFactor(0, 0)
        p.legend = lg
    return p


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def log_x(x, *plots):
    """Plot against log10(x) with decade labels 10ⁿ (pyqtgraph's own log axis labels every
    minor tick, which is unreadable over several decades). Returns the transformed x."""
    lx = np.log10(np.asarray(x, float))
    lo, hi = int(np.floor(lx.min())), int(np.ceil(lx.max()))
    major = [(k, f"10{str(k).translate(_SUP)}") for k in range(lo, hi + 1)]
    minor = [(k + np.log10(m), "") for k in range(lo, hi) for m in range(2, 10)]
    for q in plots:
        q.getAxis("bottom").setTicks([major, minor])
    return lx


def no_si(plot):
    for ax in ("left", "bottom"):
        plot.getAxis(ax).enableAutoSIPrefix(False)


def band(plot, x, lo, hi, name, alpha=38):
    c1 = pg.PlotDataItem(x, lo, pen=None)
    c2 = pg.PlotDataItem(x, hi, pen=None)
    f = pg.FillBetweenItem(c1, c2, brush=pg.mkBrush(theme.qcolor(name, alpha)))
    plot.addItem(f)
    return f


def hband(plot, lo, hi, alpha=40):
    c = QtGui.QColor(theme.current["band"]); c.setAlpha(alpha)
    r = pg.LinearRegionItem(values=(lo, hi), orientation="horizontal", movable=False,
                            brush=pg.mkBrush(c), pen=pg.mkPen(None))
    r.setZValue(-10)
    plot.addItem(r)
    return r


def message(glw, text):
    glw.clear()
    lab = glw.addLabel(text, color=theme.current["muted"], size="10pt")
    return lab


def ellipse(m, P, nsig=2.0, n=72):
    w, V = np.linalg.eigh(0.5 * (P + P.T))
    w = np.sqrt(np.maximum(w, 0.0))
    t = np.linspace(0, 2 * np.pi, n)
    pts = (V * (nsig * w)) @ np.vstack([np.cos(t), np.sin(t)])
    return m[0] + pts[0], m[1] + pts[1]


def bar_ticks(plot, names):
    plot.getAxis("bottom").setTicks([[(i, n) for i, n in enumerate(names)]])


def clamp_view(plot, pad=0.04):
    """Stop zooming / panning out past the data: the view may not leave the bounds of the
    plotted items (plus a small margin). Zooming in is unrestricted. Re-applied after every
    redraw, so the limits follow the data while it updates live."""
    vb = plot.getViewBox()
    try:
        (x0, x1), (y0, y1) = vb.childrenBounds()
    except (TypeError, ValueError):
        return
    if None in (x0, x1, y0, y1) or not np.all(np.isfinite([x0, x1, y0, y1])):
        return
    w, h = x1 - x0, y1 - y0
    w = w if w > 0 else max(abs(x0) * 0.1, 1.0)
    h = h if h > 0 else max(abs(y0) * 0.1, 1.0)
    if vb.state.get("aspectLocked"):
        # equal axis scales: the visible box must be able to take the widget's shape, so allow
        # a square of 3× the larger data extent around the data (wide or tall windows up to 3:1)
        cx, cy, r = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 1.5 * max(w, h) * (1 + 2 * pad)
        vb.setLimits(xMin=cx - r, xMax=cx + r, yMin=cy - r, yMax=cy + r,
                     maxXRange=2 * r, maxYRange=2 * r)
        return
    vb.setLimits(xMin=x0 - pad * w, xMax=x1 + pad * w, yMin=y0 - pad * h, yMax=y1 + pad * h,
                 maxXRange=w * (1 + 2 * pad), maxYRange=h * (1 + 2 * pad))


def limit_zoom(glw, pad=0.04):
    def walk(layout):
        for item in list(layout.items):
            if isinstance(item, pg.PlotItem):
                clamp_view(item, pad)
            elif isinstance(item, pg.GraphicsLayout):
                walk(item)
    walk(glw.ci)


class DataSource(QtWidgets.QWidget):
    """Simulated data, or a file of your own. `kind` is "series" (time series) or "static"
    (inputs, y[, σ]); the page supplies `reader(path)` for static files, because the
    input columns depend on the selected problem."""
    changed = QtCore.pyqtSignal()
    FILTER = "Data files (*.csv *.tsv *.txt *.dat *.npy *.npz);;All files (*)"

    def __init__(self, kind="series", reader=None, parent=None):
        super().__init__(parent)
        self.kind, self.reader = kind, reader
        self.path = self.data = None
        v = QtWidgets.QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(6)
        h = QtWidgets.QHBoxLayout()
        self.rb_sim = QtWidgets.QRadioButton("Simulated"); self.rb_sim.setChecked(True)
        self.rb_file = QtWidgets.QRadioButton("My data"); self.rb_file.setEnabled(False)
        self.b_load = QtWidgets.QPushButton("Load file…")
        h.addWidget(self.rb_sim); h.addWidget(self.rb_file); h.addStretch(1); h.addWidget(self.b_load)
        v.addLayout(h)
        self.info = muted("" if kind == "series" else ""); v.addWidget(self.info)
        self.b_load.clicked.connect(self._browse)
        self.rb_file.toggled.connect(lambda _: self.changed.emit())

    def from_file(self):
        return self.rb_file.isChecked() and self.data is not None

    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load data", "", self.FILTER)
        if path:
            self.load(path)

    def load(self, path):
        """Read `path`; on success switch to "My data". Returns True on success."""
        try:
            data = self._read(path)
        except Exception as e:                       # bad file: say why, keep the old data
            self.info.setText(f"could not read {path.split('/')[-1]}: {e}")
            return False
        self.path, self.data = path, data
        self.rb_file.setEnabled(True)
        self._describe()
        if self.rb_file.isChecked():
            self.changed.emit()
        else:
            self.rb_file.setChecked(True)            # emits changed
        return True

    def reload(self):
        """Re-read the current file (static data: after the problem, i.e. the input columns,
        changed). Returns False if the file does not fit."""
        if self.path is None:
            return False
        try:
            self.data = self._read(self.path)
        except Exception as e:
            self.info.setText(str(e))
            return False
        self._describe()
        return True

    def _read(self, path):
        if self.kind == "series":
            from ..core.userdata import load_series
            return load_series(path)
        return self.reader(path)

    def _describe(self):
        name = str(self.path).replace("\\", "/").split("/")[-1]
        txt = f"{name}: {self.data.describe()}"
        for w in getattr(self.data, "warnings", []):
            txt += f"\n⚠ {w}"
        self.info.setText(txt)


class PlotView(QtWidgets.QWidget):
    """A view = optional header row + one GraphicsLayoutWidget, redrawn lazily.

    Reusable views keep their PlotItems between redraws while `begin(key)` sees the same
    layout key: only the data items are replaced, which is ~3× faster than rebuilding the
    plots and keeps the user's zoom while estimates update live."""

    def __init__(self, title, draw, parent=None, reusable=False):
        super().__init__(parent)
        self.title, self.draw_fn, self.reusable = title, draw, reusable
        self.dirty = True
        self._key, self._plots = None, None
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6); lay.setSpacing(4)
        self.header = QtWidgets.QHBoxLayout(); self.header.setSpacing(8)
        lay.addLayout(self.header)
        self.glw = pg.GraphicsLayoutWidget()
        lay.addWidget(self.glw, 1)

    def begin(self, key):
        """Return the cached plots (cleared of data) if the layout key is unchanged, else
        clear the canvas and return None (the caller builds plots and calls keep())."""
        if self._plots is not None and key == self._key:
            for p in self._plots:
                p.clear()
            return self._plots
        self.glw.clear()
        self._key, self._plots = key, None
        return None

    def keep(self, plots):
        self._plots = list(plots)

    def redraw(self):
        self.glw.setBackground(theme.current["plot_bg"])
        if not self.reusable:
            self.glw.clear()
        try:
            self.draw_fn(self.glw)
        except _Empty as e:
            self._key = self._plots = None
            message(self.glw, str(e))
        except Exception as e:  # keep the GUI alive, show why
            self._key = self._plots = None
            message(self.glw, f"cannot draw: {type(e).__name__}: {e}")
            traceback.print_exc()
        limit_zoom(self.glw)
        self.dirty = False


class _Empty(Exception):
    """Raised by a draw function when there is nothing to show yet (message shown instead)."""


Empty = _Empty
