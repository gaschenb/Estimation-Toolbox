"""Estimation Toolbox desktop app (PyQt5 + pyqtgraph).

    esttoolbox-gui      or      python -m esttoolbox.gui

Three workspaces in the sidebar:
  State estimation    live filtering / smoothing / MHE: drag a parameter slider and the data
                      are re-simulated and every estimator re-runs in the background; linked
                      time cursor, track with covariance ellipses, NEES/NIS, innovations,
                      particle densities, Monte Carlo consistency
  System ID           PEM (KF/EKF/UKF/CKF likelihood), EM, PMMH with likelihood slices
  Static parameters   12 problems; LS / MLE / MAP / Fisher / profile / bootstrap live, and four
                      MCMC samplers with corner plots, traces and R̂ / ESS

Computation runs in worker threads; a newer request cancels an older one and stale results are
discarded, so the interface never blocks. Plots are pyqtgraph (GPU-friendly, pan/zoom with the
mouse, right-click for export and axis options).
"""
from __future__ import annotations

import os
import sys

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

from . import theme

APP_NAME = "State Estimation Toolbox"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, theme_name="light"):
        super().__init__()
        app = QtWidgets.QApplication.instance()
        theme.apply(app, theme_name)
        from .state_page import StatePage          # import after the theme is configured
        from .static_page import StaticPage
        from .sysid_page import SysIdPage
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 920)

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        h = QtWidgets.QHBoxLayout(central); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(0)
        side = QtWidgets.QWidget(); side.setFixedWidth(196)
        sv = QtWidgets.QVBoxLayout(side); sv.setContentsMargins(0, 0, 0, 0); sv.setSpacing(0)
        b = QtWidgets.QLabel(APP_NAME); b.setObjectName("brand"); b.setWordWrap(True)
        sv.addWidget(b)
        s = QtWidgets.QLabel("estimation & inference"); s.setObjectName("brandSub")
        sv.addWidget(s)
        self.nav = QtWidgets.QListWidget(); self.nav.setObjectName("nav")
        self.nav.setFocusPolicy(Qt.NoFocus)
        for name in ("State estimation", "System identification", "Static parameters"):
            self.nav.addItem(name)
        sv.addWidget(self.nav, 1)
        h.addWidget(side)
        self.stack = QtWidgets.QStackedWidget()
        h.addWidget(self.stack, 1)
        self.state, self.sysid, self.static = StatePage(), SysIdPage(), StaticPage()
        self.pages = [self.state, self.sysid, self.static]
        for p in self.pages:
            self.stack.addWidget(p)
            p.status.connect(self.show_status)
        self.nav.currentRowChanged.connect(self._page_changed)

        tb = QtWidgets.QToolBar(); tb.setMovable(False); tb.setIconSize(QtCore.QSize(16, 16))
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(tb)
        self.a_run = tb.addAction("▶  Run", self.run); self.a_run.setShortcut("Ctrl+R")
        self.a_cancel = tb.addAction("■  Cancel", self.cancel); self.a_cancel.setShortcut("Esc")
        self.a_live = tb.addAction("Live updates"); self.a_live.setCheckable(True)
        self.a_live.setChecked(True); self.a_live.setShortcut("Ctrl+L")
        self.a_live.setToolTip("Re-run fast estimators whenever a control changes (Ctrl+L)")
        self.a_live.toggled.connect(self._live)
        tb.addSeparator()
        self.a_mc = tb.addAction("Monte Carlo", lambda: self.state.run_mc()); self.a_mc.setShortcut("Ctrl+M")
        tb.addSeparator()
        tb.addAction("Export plot…", self.export_png).setShortcut("Ctrl+E")
        tb.addAction("Export table…", self.export_csv).setShortcut("Ctrl+Shift+E")
        spacer = QtWidgets.QWidget(); spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                                          QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(spacer)
        self.a_theme = tb.addAction("Dark theme", self.toggle_theme); self.a_theme.setShortcut("Ctrl+T")
        tb.addAction("Help", self.about).setShortcut("F1")

        self.busy = QtWidgets.QProgressBar(); self.busy.setRange(0, 0); self.busy.setFixedWidth(120)
        self.busy.setVisible(False)
        self.statusBar().addPermanentWidget(self.busy)
        for p in self.pages:
            for r in [getattr(p, "runner", None), getattr(p, "mc_runner", None)]:
                if r is not None:
                    r.busy.connect(self._busy)
        self.nav.setCurrentRow(0)

    def _busy(self, _):
        runners = [r for p in self.pages for r in (getattr(p, "runner", None),
                                                    getattr(p, "mc_runner", None)) if r is not None]
        self.busy.setVisible(any(r.running() for r in runners))

    def page(self):
        return self.stack.currentWidget()

    def _page_changed(self, i):
        self.stack.setCurrentIndex(i)
        self.a_mc.setEnabled(i == 0)
        self.a_live.setEnabled(i != 1)
        self.page().refresh()

    def show_status(self, msg):
        self.statusBar().showMessage(msg)

    def run(self):
        self.page().run_all()

    def cancel(self):
        self.page().cancel()

    def _live(self, on):
        for p in self.pages:
            p.set_live(on)

    def export_png(self):
        v = self.page().current_view()
        name = getattr(v, "title", "plot").lower().replace(" ", "_")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export plot", f"{name}.png",
                                                        "PNG image (*.png)")
        if path:
            self.page().export_png(path); self.show_status(f"saved {path}")

    def export_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export results", "results.csv",
                                                        "CSV (*.csv)")
        if path:
            self.page().export_csv(path); self.show_status(f"saved {path}")

    def toggle_theme(self):
        new = "dark" if theme.current["name"] == "light" else "light"
        theme.apply(QtWidgets.QApplication.instance(), new)
        self.a_theme.setText("Light theme" if new == "dark" else "Dark theme")
        for p in self.pages:
            p.invalidate()
            if getattr(p, "table", None) is not None and hasattr(p, "_refresh_table"):
                p._refresh_table()

    def about(self):
        QtWidgets.QMessageBox.information(self, APP_NAME, (
            f"{APP_NAME}\n\n"
            "Ctrl+R  run everything (including slow methods)\n"
            "Esc     cancel the running job\n"
            "Ctrl+L  toggle live updates\n"
            "Ctrl+M  Monte Carlo study (state estimation)\n"
            "Ctrl+E  export the current plot, Ctrl+Shift+E the table\n"
            "Ctrl+T  light / dark theme\n\n"
            "Plots: drag to pan, wheel to zoom, right-click for options, click a legend entry "
            "to hide a curve. State plots: click or drag the cursor, ←/→ step (Shift: ×10). "
            "Double-click a row in the results table to show only that method."))

    def closeEvent(self, e):
        for p in self.pages:
            for r in (getattr(p, "runner", None), getattr(p, "mc_runner", None)):
                if r is not None:
                    r.cancel()
                    for t in list(r.threads):
                        t.wait(3000)
        super().closeEvent(e)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    QtCore.QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QtCore.QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    app.setApplicationName(APP_NAME)
    f = app.font(); f.setPointSize(9)
    if sys.platform.startswith("win"):
        f.setFamily("Segoe UI")
    app.setFont(f)
    w = MainWindow("dark" if "--dark" in argv or os.environ.get("ESTTOOLBOX_THEME") == "dark"
                   else "light")
    w.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
