"""Your own data: loading measurement files and building models around them.

Time series (state estimation, system identification)
    A table with one row per time step. Column names decide the role of each column:
        t or time            sample times (optional; uniform spacing expected)
        x_<name>, true_<…>   ground truth of a state (optional; enables RMSE / NEES)
        anything else        a measurement channel
    Without a header every column is a measurement. Files: .csv, .tsv, .txt, .npy, .npz
    (npz keys: y, and optionally t, x).

Static data (curve fitting)
    A table with the model inputs first (as named by the problem, e.g. t, or S and I),
    then y, then optionally sigma (per-point standard deviation).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TIME_NAMES = ("t", "time", "timestamp", "seconds", "sec", "s")
TRUTH_PREFIXES = ("x_", "true_", "truth_")


# ----------------------------------------------------------------------------- reading
def read_table(path):
    """Read a numeric table. Returns (column names or None, (n, k) float array)."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".npy":
        a = np.load(path)
        return None, np.atleast_2d(a.T).T.astype(float) if a.ndim == 1 else a.astype(float)
    if suf == ".npz":
        z = np.load(path)
        names, cols = [], []
        for k in z.files:
            a = np.asarray(z[k], float)
            a = a[:, None] if a.ndim == 1 else a
            for j in range(a.shape[1]):
                names.append(k if a.shape[1] == 1 else f"{k}{j + 1}")
                cols.append(a[:, j])
        n = min(len(c) for c in cols)
        return _npz_names(names), np.column_stack([c[:n] for c in cols])
    text = path.read_text(encoding="utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"{path.name}: no data")
    try:
        delim = csv.Sniffer().sniff(lines[0], delimiters=",;\t ").delimiter
    except csv.Error:
        delim = ","
    rows = list(csv.reader(lines, delimiter=delim, skipinitialspace=True))
    rows = [[c for c in r if c != ""] if delim == " " else r for r in rows]
    header = None
    try:
        [float(c) for c in rows[0]]
    except ValueError:
        header = [c.strip() for c in rows[0]]
        rows = rows[1:]
    k = len(rows[0])
    data = np.full((len(rows), k), np.nan)
    for i, r in enumerate(rows):
        if len(r) != k:
            raise ValueError(f"{path.name}: row {i + 2 if header else i + 1} has {len(r)} "
                             f"columns, expected {k}")
        for j, c in enumerate(r):
            c = c.strip()
            data[i, j] = float(c) if c not in ("", "nan", "NaN", "NA") else np.nan
    return header, data


def _npz_names(names):
    out = []
    for n in names:
        if n.startswith("y"):
            out.append(n if n != "y" else "y1")
        elif n.startswith("x"):
            out.append("x_" + (n[1:] or "1"))
        else:
            out.append(n)
    return out


# ----------------------------------------------------------------------------- time series
@dataclass
class SeriesData:
    ys: np.ndarray                    # (T, ny) measurements
    t: np.ndarray                     # (T,) sample times
    dt: float
    obs_names: list
    xs: np.ndarray | None = None      # (T, nx_truth) ground truth at the sample times
    truth_names: list = field(default_factory=list)
    source: str = ""
    warnings: list = field(default_factory=list)

    @property
    def T(self):
        return len(self.ys)

    @property
    def ny(self):
        return self.ys.shape[1]

    def describe(self):
        s = f"{self.T} samples × {self.ny} channels ({', '.join(self.obs_names)}), Δt = {self.dt:.4g}"
        if self.xs is not None:
            s += f"; truth for {', '.join(self.truth_names)}"
        return s


def load_series(path):
    """Read a measurement time series (see module docstring for the file layout)."""
    header, a = read_table(path)
    k = a.shape[1]
    names = header or [f"y{j + 1}" for j in range(k)]
    low = [n.lower() for n in names]
    t_col = next((j for j, n in enumerate(low) if n in TIME_NAMES), None)
    truth = [j for j, n in enumerate(low) if n.startswith(TRUTH_PREFIXES)]
    meas = [j for j in range(k) if j != t_col and j not in truth]
    if not meas:
        raise ValueError("no measurement columns found")
    bad = ~np.all(np.isfinite(a[:, meas]), axis=1)
    warnings = []
    if bad.any():
        warnings.append(f"dropped {int(bad.sum())} rows with missing measurements")
        a = a[~bad]
    if len(a) < 3:
        raise ValueError("need at least 3 samples")
    if t_col is not None:
        t = a[:, t_col]
        if np.any(np.diff(t) <= 0):
            order = np.argsort(t, kind="stable")
            a, t = a[order], t[order]
            if np.any(np.diff(t) <= 0):
                raise ValueError("time column has repeated values")
            warnings.append("rows were sorted by time")
        d = np.diff(t)
        dt = float(np.median(d))
        if np.max(np.abs(d - dt)) > 0.05 * dt:
            warnings.append(f"sampling is not uniform (Δt from {d.min():.4g} to {d.max():.4g}); "
                            f"the models assume Δt = {dt:.4g}")
    else:
        dt = 1.0
        t = np.arange(1, len(a) + 1, dtype=float)
    xs = a[:, truth] if truth else None
    tn = [names[j].split("_", 1)[1] if "_" in names[j] else names[j] for j in truth]
    return SeriesData(a[:, meas], t, dt, [names[j] for j in meas], xs, tn, str(path), warnings)


def series_from_arrays(ys, t=None, xs=None, obs_names=None):
    ys = np.asarray(ys, float)
    ys = ys[:, None] if ys.ndim == 1 else ys
    if t is None:
        t = np.arange(1, len(ys) + 1, dtype=float); dt = 1.0
    else:
        t = np.asarray(t, float); dt = float(np.median(np.diff(t)))
    xs = None if xs is None else np.asarray(xs, float).reshape(len(ys), -1)
    names = obs_names or [f"y{j + 1}" for j in range(ys.shape[1])]
    tn = [f"x{j + 1}" for j in range(xs.shape[1])] if xs is not None else []
    return SeriesData(ys, t, dt, list(names), xs, tn)


def full_truth(data, model):
    """Ground truth in the model's state layout: (T+1, nx), row 0 (the prior time) NaN, and NaN
    columns for states the file has no truth for. Columns are matched by state name, or by
    position when the file has exactly nx truth columns. None without any match."""
    if data.xs is None:
        return None
    names = list(model.state_names)
    X = np.full((data.T + 1, model.nx), np.nan)
    hit = False
    for i, n in enumerate(names):
        if n in data.truth_names:
            X[1:, i] = data.xs[:, data.truth_names.index(n)]; hit = True
    if not hit and data.xs.shape[1] == model.nx:
        X[1:] = data.xs; hit = True
    return X if hit else None


# ----------------------------------------------------------------------------- kinematic models
KINEMATIC = {"rw": ("random walk", 1), "cv": ("constant velocity", 2),
             "ca": ("constant acceleration", 3)}


def _noise_guess(ys, dt, order):
    """Split the variance of the order-th differences between measurement and process noise.
    For y = position + white noise of std r and a white (order)-th derivative of intensity q:
    Var Δⁿy = C(2n, n) r² + c_n q dt^{2n−1}."""
    from math import comb
    d = np.diff(ys, n=order, axis=0)
    v = np.median(np.abs(d - np.median(d, 0)), 0) ** 2 * 2.198     # robust variance (MAD²·1.4826²)
    v = float(np.mean(np.maximum(v, 1e-300)))
    cn = {1: 1.0, 2: 2.0 / 3.0, 3: 11.0 / 20.0}[order]
    r = np.sqrt(0.5 * v / comb(2 * order, order))
    q = 0.5 * v / (cn * dt ** (2 * order - 1))
    return max(q, 1e-12), max(r, 1e-12)


def kinematic_problem(data, order="cv", name=None):
    """Nearly-constant random walk / velocity / acceleration model in as many dimensions as the
    data have channels, positions measured directly. θ = (q, r): white-noise intensity of the
    highest derivative and measurement standard deviation (Bar-Shalom et al. 2001, §6)."""
    from .linalg import van_loan
    from .ssm import LinearGaussianSSM
    from .systems import SSMProblem
    label, n = KINEMATIC[order]
    d, dt = data.ny, data.dt
    y0 = data.ys[0]
    span = np.ptp(data.ys, axis=0) + 1e-12
    q0, r0 = _noise_guess(data.ys, dt, n)
    obs = list(data.obs_names)
    deriv = ["", "v", "a"]
    snames = [(f"{deriv[o]}_{c}" if o else c) for o in range(n) for c in obs]

    def make(theta, d=d, dt=dt, n=n):
        q, r = float(theta[0]), float(theta[1])
        A1 = np.diag(np.ones(n - 1), 1)
        B1 = np.zeros((n, 1)); B1[-1, 0] = 1.0
        F1, Q1 = van_loan(A1, B1, np.array([[q]]), dt)
        A = np.kron(F1, np.eye(d)); Q = np.kron(Q1, np.eye(d))
        C = np.kron(np.eye(1, n), np.eye(d))
        m0 = np.zeros(n * d); m0[:d] = y0
        p0 = [np.full(d, r ** 2 * 4 + 1e-12)]
        T = max(data.t[-1] - data.t[0], dt)
        for o in range(1, n):
            p0.append((span / T ** o) ** 2 * 4)
        return LinearGaussianSSM(f"{order}-{d}d", A, C, Q, r ** 2 * np.eye(d), m0=m0,
                                 P0=np.diag(np.concatenate(p0)), state_names=snames,
                                 obs_names=obs, dt=dt,
                                 description=f"nearly-{label} model, {d} measured "
                                             f"channel{'s' if d > 1 else ''}")
    th = np.array([q0, r0])
    return SSMProblem(
        name or f"{label} ({d}-D)", f"Nearly-{label} kinematic model fitted around your data", make,
        ["q", "r"], th, th * 1e-4, th * 1e4, T=data.T, prior_mean=th.copy(),
        prior_std=np.array([2.0, 2.0]), xy=(0, 1) if d >= 2 else None,
        recommended=("kf", "srkf", "ukf"), meas_xy=(lambda ys: ys[:, :2]) if d >= 2 else None,
        category="your data")


def data_problems(data):
    """Models that can explain a measurement file: the three kinematic models plus every
    built-in system with the same number of measurement channels."""
    from .systems import list_ssm
    out = {f"{lbl} ({data.ny}-D)": kinematic_problem(data, k) for k, (lbl, _) in KINEMATIC.items()}
    for name, p in list_ssm().items():
        if p.model().ny == data.ny:
            out[name] = p
    return out


def problem_spec(problem, data):
    """Picklable description of a data-driven problem, for worker processes."""
    for k, (lbl, _) in KINEMATIC.items():
        if problem.name.startswith(lbl):
            return ("kinematic", k, data.ys, data.t, data.obs_names)
    return problem.name


def resolve_problem(spec):
    from .systems import get_ssm
    if isinstance(spec, tuple) and spec[0] == "kinematic":
        _, order, ys, t, names = spec
        return kinematic_problem(series_from_arrays(ys, t, obs_names=names), order)
    return get_ssm(spec)


# ----------------------------------------------------------------------------- static data
@dataclass
class StaticData:
    X: np.ndarray                     # (n, k) model inputs
    y: np.ndarray                     # (n,)
    sigma: np.ndarray | None = None   # (n,) per-point standard deviations
    input_names: list = field(default_factory=list)
    source: str = ""

    def describe(self):
        s = f"{len(self.y)} points, inputs {', '.join(self.input_names)}"
        return s + (", per-point σ" if self.sigma is not None else "")


def load_static(path, inputs=None):
    """Read (inputs…, y[, sigma]). With a header, columns are matched by name ("y", "sigma" /
    "sd" / "err"); without one, the last column is y, or y then sigma when `inputs` says how
    many input columns there are."""
    header, a = read_table(path)
    k = a.shape[1]
    if k < 2:
        raise ValueError("need at least an input column and a y column")
    low = [h.lower() for h in header] if header else None
    s_names = ("sigma", "sd", "std", "err", "error", "dy", "uncertainty")
    if low and "y" in low:
        jy = low.index("y")
        js = next((j for j, n in enumerate(low) if n in s_names), None)
        jx = [j for j in range(k) if j not in (jy, js)]
    else:
        ni = len(inputs) if inputs else k - 1
        jx = list(range(ni)); jy = ni
        js = ni + 1 if k > ni + 1 else None
    if inputs:
        if low:
            jx = [low.index(n.lower()) if n.lower() in low else jx[i]
                  for i, n in enumerate(inputs[:len(jx)])]
        if len(jx) < len(inputs):
            raise ValueError(f"this problem needs {len(inputs)} input column(s) "
                             f"({', '.join(inputs)}); the file has {len(jx)}")
        jx = jx[:len(inputs)]
    X = a[:, jx]; y = a[:, jy]; s = a[:, js] if js is not None else None
    ok = np.all(np.isfinite(X), 1) & np.isfinite(y) & (np.isfinite(s) & (s > 0) if s is not None else True)
    names = [header[j] for j in jx] if header else (inputs or [f"x{j + 1}" for j in range(len(jx))])
    return StaticData(X[ok], y[ok], None if s is None else s[ok], list(names), str(path))


def noise_guess(X, y):
    """Robust σ from neighbouring points (sorted by the first input): MAD of first differences."""
    o = np.lexsort(X.T[::-1])
    d = np.diff(y[o])
    return float(max(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2), 1e-12))
