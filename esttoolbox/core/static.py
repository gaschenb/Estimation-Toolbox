"""Static (non-dynamic) parameter-estimation problems  y = f(θ) + ε,  ε ~ N(0, σ²I).

Twelve problems (oscillator, heat, projectile, reaction–diffusion, optical blur, acoustic
source, enzyme kinetics, pharmacokinetics, spectroscopy, exoplanet radial velocity, SIR
epidemic, Theis well test) with batched forward models:
forward(Θ) accepts Θ of shape (p,) or (B, p) and returns (n,) or (B, n), so finite-
difference Jacobians (2p points), ensemble MCMC moves and multi-chain proposals are
single vectorized calls.

The reaction–diffusion model uses exact spectral (DCT-I) diffusion with Strang-split exact
logistic reaction on a fixed step, so f is smooth in (D, r) (an explicit scheme whose step
count depends on D would make f piecewise in D); the projectile RK4 and the PSF convolution
are vectorized across Θ.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.fft import dct, idct


@dataclass
class StaticProblem:
    name: str
    description: str
    param_names: list
    theta_true: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    prior_mean: np.ndarray
    prior_std: np.ndarray
    noise_std: float
    _forward: callable = field(repr=False)
    plot_kind: str = "series"
    design: object = field(default=None, repr=False)
    x: np.ndarray = field(default=None, repr=False)        # abscissa of each observation
    groups: np.ndarray = field(default=None, repr=False)   # curve label of each observation
    xlabel: str = "observation index"
    xlog: bool = False
    category: str = "physics"
    inputs: tuple = ()                  # names of the model inputs, for your own data
    xfun: callable = field(default=None, repr=False)   # (Θ, X) -> (B, n) at arbitrary inputs
    guess: callable = field(default=None, repr=False)  # (X, y) -> (θ₀, lo, hi) from the data
    has_truth: bool = True              # False for measured data: θ_true is only a start value
    scale: np.ndarray = field(default=None, repr=False)  # per-point σ (data are whitened by it)

    @property
    def dim(self):
        return len(self.param_names)

    def forward(self, theta):
        th = np.asarray(theta, float)
        single = th.ndim == 1
        Th = np.minimum(np.maximum(th.reshape(-1, th.shape[-1]), self.lo), self.hi)
        Y = self._forward(Th)
        return Y[0] if single else Y

    def simulate(self, noise_std=None, rng=None, theta=None):
        rng = np.random.default_rng(rng)
        s = self.noise_std if noise_std is None else float(noise_std)
        y = self.forward(self.theta_true if theta is None else theta)
        return y + s * rng.standard_normal(y.shape), s

    @property
    def accepts_data(self):
        return self.xfun is not None

    def with_data(self, data, theta0=None):
        """This model fitted to measured data (userdata.StaticData). Returns (problem, y, σ).
        With per-point σ the residuals are whitened (y/σᵢ, f/σᵢ, σ = 1); `scale` undoes it
        for plotting."""
        from .userdata import noise_guess
        if not self.accepts_data:
            raise ValueError(f"'{self.name}' has a fixed design and cannot take your data")
        X = np.asarray(data.X, float).reshape(len(data.y), -1)[:, :len(self.inputs)]
        y = np.asarray(data.y, float)
        f = self.xfun
        if self.guess is not None:
            th0, lo, hi = (np.asarray(a, float) for a in self.guess(X, y))
            pstd = np.maximum(np.abs(hi - lo) / 4, 1e-12)
        else:
            th0, lo, hi, pstd = self.prior_mean, self.lo, self.hi, self.prior_std
        if theta0 is not None:
            th0 = np.clip(np.asarray(theta0, float), lo, hi)
        s = None if data.sigma is None else np.asarray(data.sigma, float)
        if s is not None:
            fwd = lambda Th: f(Th, X) / s  # noqa: E731
            yw, sig = y / s, 1.0
        else:
            fwd = lambda Th: f(Th, X)  # noqa: E731
            yw, sig = y, noise_guess(X, y)
        groups = None
        if X.shape[1] > 1:
            nm = (list(getattr(data, "input_names", [])) + list(self.inputs))[1] \
                if len(getattr(data, "input_names", [])) > 1 else self.inputs[1]
            groups = np.array([f"{nm} = {v:g}" for v in X[:, 1]])
        p = replace(self, _forward=fwd, theta_true=th0, lo=lo, hi=hi, prior_mean=th0.copy(),
                    prior_std=pstd, noise_std=sig, plot_kind="series", design=X, x=X[:, 0],
                    groups=groups, xlabel=(list(getattr(data, "input_names", [])) or
                                            list(self.inputs))[0],
                    has_truth=False, scale=s)
        return p, yw, sig


_STATIC: dict[str, StaticProblem] = {}


def register(p):
    _STATIC[p.name] = p
    return p


def get_static(name):
    try:
        return _STATIC[name]
    except KeyError:
        raise KeyError(f"unknown static problem '{name}'; available: {', '.join(_STATIC)}")


def list_static():
    return dict(_STATIC)


# --- damped oscillator --------------------------------------------------------------
_t_osc = np.linspace(0.0, 10.0, 200)


def _osc_x(Th, X):
    t = X[:, 0]
    A, z, w0, phi = (Th[:, i:i + 1] for i in range(4))
    wd = w0 * np.sqrt(np.maximum(1.0 - z**2, 1e-12))
    return A * np.exp(-z * w0 * t) * np.cos(wd * t + phi)


def _osc(Th):
    return _osc_x(Th, _t_osc[:, None])


register(StaticProblem(
    "oscillator", "Damped harmonic oscillator, y(t) = A e^{−ζω₀t} cos(ω_d t + φ)",
    ["A", "zeta", "omega0", "phi"], np.array([1.5, 0.10, 4.0, 0.5]),
    np.array([0.1, 0.001, 0.5, -np.pi]), np.array([5.0, 0.9, 20.0, np.pi]),
    np.array([1.0, 0.2, 3.0, 0.0]), np.array([1.0, 0.2, 2.0, 1.0]), 0.08, _osc,
    design=_t_osc, x=_t_osc, xlabel="time [s]", category="mechanics", inputs=("t",), xfun=_osc_x))

# --- heat pulse -----------------------------------------------------------------------
_x_heat = np.linspace(0.0, 10.0, 60)
_t_heat = np.array([0.5, 2.0, 5.0])


def _heat(Th):
    a, Q, x0 = (Th[:, i, None, None] for i in range(3))
    t = _t_heat[None, :, None]
    u = Q / np.sqrt(4 * np.pi * a * t) * np.exp(-(_x_heat[None, None] - x0) ** 2 / (4 * a * t))
    return u.reshape(len(Th), -1)


register(StaticProblem(
    "heat", "1-D heat equation: diffusing Gaussian pulse observed at t = 0.5, 2, 5",
    ["alpha", "Q", "x0"], np.array([0.8, 5.0, 4.0]), np.array([0.05, 0.5, 0.0]),
    np.array([5.0, 20.0, 10.0]), np.array([1.0, 4.0, 5.0]), np.array([1.0, 3.0, 2.0]),
    0.05, _heat, design=(_x_heat, _t_heat), x=np.tile(_x_heat, 3),
    groups=np.repeat([f"t = {t:g}" for t in _t_heat], len(_x_heat)), xlabel="x",
    category="heat transfer"))

# --- projectile with quadratic drag -----------------------------------------------------
_t_proj = np.linspace(0.0, 2.0, 40)


def _proj_scalar(v0, ang_deg, c):
    """Pure-float RK4 for one θ: ~20× cheaper than array code at batch size 1."""
    import math
    g = 9.81
    a = math.radians(ang_deg)
    x = y = 0.0
    u, w = v0 * math.cos(a), v0 * math.sin(a)
    dt = float(_t_proj[1] - _t_proj[0])
    out = [0.0, 0.0]
    for _ in range(1, len(_t_proj)):
        s = c * math.sqrt(u * u + w * w); a1u, a1w = -s * u, -g - s * w
        u2, w2 = u + 0.5 * dt * a1u, w + 0.5 * dt * a1w
        s = c * math.sqrt(u2 * u2 + w2 * w2); a2u, a2w = -s * u2, -g - s * w2
        u3, w3 = u + 0.5 * dt * a2u, w + 0.5 * dt * a2w
        s = c * math.sqrt(u3 * u3 + w3 * w3); a3u, a3w = -s * u3, -g - s * w3
        u4, w4 = u + dt * a3u, w + dt * a3w
        s = c * math.sqrt(u4 * u4 + w4 * w4); a4u, a4w = -s * u4, -g - s * w4
        x += dt / 6 * (u + 2 * u2 + 2 * u3 + u4)
        y += dt / 6 * (w + 2 * w2 + 2 * w3 + w4)
        u += dt / 6 * (a1u + 2 * a2u + 2 * a3u + a4u)
        w += dt / 6 * (a1w + 2 * a2w + 2 * a3w + a4w)
        out += (x, y)
    return np.array(out)


def _proj(Th):
    """RK4 on (x, y, vx, vy) with component arrays (B,) — no per-step stacking."""
    if len(Th) <= 2:
        return np.stack([_proj_scalar(*th) for th in Th.tolist()])
    v0, ang, c = Th[:, 0], np.deg2rad(Th[:, 1]), Th[:, 2]
    B = len(Th)
    g = 9.81
    x = np.zeros(B); y = np.zeros(B)
    u, w = v0 * np.cos(ang), v0 * np.sin(ang)
    dt = _t_proj[1] - _t_proj[0]
    out = np.zeros((B, len(_t_proj), 2))

    def acc(u, w):
        s = c * np.sqrt(u * u + w * w)
        return -s * u, -g - s * w

    for i in range(1, len(_t_proj)):
        a1u, a1w = acc(u, w)
        u2, w2 = u + 0.5 * dt * a1u, w + 0.5 * dt * a1w
        a2u, a2w = acc(u2, w2)
        u3, w3 = u + 0.5 * dt * a2u, w + 0.5 * dt * a2w
        a3u, a3w = acc(u3, w3)
        u4, w4 = u + dt * a3u, w + dt * a3w
        a4u, a4w = acc(u4, w4)
        x = x + dt / 6 * (u + 2 * u2 + 2 * u3 + u4)
        y = y + dt / 6 * (w + 2 * w2 + 2 * w3 + w4)
        u = u + dt / 6 * (a1u + 2 * a2u + 2 * a3u + a4u)
        w = w + dt / 6 * (a1w + 2 * a2w + 2 * a3w + a4w)
        out[:, i, 0], out[:, i, 1] = x, y
    return out.reshape(B, -1)


register(StaticProblem(
    "projectile", "Projectile with quadratic drag, RK4 (x, y) trajectory",
    ["v0", "angle_deg", "c_drag"], np.array([20.0, 45.0, 0.05]), np.array([5.0, 10.0, 0.0]),
    np.array([60.0, 80.0, 0.5]), np.array([15.0, 40.0, 0.1]), np.array([10.0, 15.0, 0.1]),
    0.15, _proj, plot_kind="xy", design=_t_proj, category="mechanics"))

# --- Fisher–KPP reaction–diffusion ------------------------------------------------------
_x_rd = np.linspace(0.0, 20.0, 80)
_u0_rd = (_x_rd < 0.15 * _x_rd[-1]).astype(float)
_lam_rd = -(2.0 - 2.0 * np.cos(np.pi * np.arange(80) / 79)) / (_x_rd[1] - _x_rd[0]) ** 2


_C_rd = dct(np.eye(80), type=1, axis=0)                  # DCT-I as a matrix
_Ci_rd = idct(np.eye(80), type=1, axis=0)


def _rd(Th, T=4.0, n_steps=80):
    """u_t = D u_xx + r u(1−u), Neumann BC. Strang splitting with exact sub-flows:
    diffusion is diagonal in the DCT-I basis (Neumann FD Laplacian);
    the logistic ODE has the closed form u e^{rt}/(1 − u + u e^{rt}). Adjacent half
    reaction steps are merged, so each step is one reaction and two small matmuls."""
    D, r = Th[:, :1], Th[:, 1:2]
    dt = T / n_steps
    u = np.repeat(_u0_rd[None], len(Th), axis=0)
    eh, ef = np.exp(r * dt / 2), np.exp(r * dt)
    decay = np.exp(D * _lam_rd[None] * dt)
    CT, CiT = _C_rd.T, _Ci_rd.T
    u = u * eh / (1.0 - u + u * eh)
    for i in range(n_steps):
        u = ((u @ CT) * decay) @ CiT
        e = ef if i < n_steps - 1 else eh
        u = u * e / (1.0 - u + u * e)
    return u


register(StaticProblem(
    "reaction-diffusion", "Fisher–KPP front u(x, T): spectral diffusion + exact logistic split",
    ["D", "r"], np.array([1.0, 1.5]), np.array([0.05, 0.1]), np.array([4.0, 5.0]),
    np.array([0.8, 1.0]), np.array([1.0, 1.0]), 0.03, _rd, design=_x_rd, x=_x_rd, xlabel="x",
    category="PDE"))

# --- optical blur -------------------------------------------------------------------------
_scene = np.zeros(160)
_scene[30:60] = 1.0
_scene[80:90] = 0.6
_scene[110:150] = np.linspace(0, 1, 40)
_pad = np.pad(_scene, 30, mode="edge")
_win = np.lib.stride_tricks.sliding_window_view(_pad, 61)[:, ::-1]   # (160, 61)
_k = np.arange(-30, 31, dtype=float)


def _blur(Th):
    s, gain = Th[:, :1], Th[:, 1:2]
    psf = np.exp(-0.5 * (_k[None] / np.maximum(s, 1e-3)) ** 2)
    psf /= psf.sum(1, keepdims=True)
    return gain * (psf @ _win.T)


register(StaticProblem(
    "optical-blur", "Known 1-D scene ⊛ Gaussian PSF × gain",
    ["sigma_psf", "gain"], np.array([3.0, 1.2]), np.array([0.2, 0.1]), np.array([15.0, 5.0]),
    np.array([2.0, 1.0]), np.array([2.0, 0.5]), 0.02, _blur, design=_scene, xlabel="pixel",
    category="imaging"))

# --- acoustic source ----------------------------------------------------------------------
_gx, _gy = np.meshgrid(np.linspace(0, 10, 7), np.linspace(0, 10, 7))
_sens = np.column_stack([_gx.ravel(), _gy.ravel()])


def _ac(Th):
    x0, y0, A, al = (Th[:, i:i + 1] for i in range(4))
    r = np.maximum(np.hypot(_sens[None, :, 0] - x0, _sens[None, :, 1] - y0), 0.3)
    return A / r * np.exp(-al * r)


register(StaticProblem(
    "acoustic", "Acoustic point source p = A e^{−αr}/r on a 7×7 sensor grid",
    ["x0", "y0", "A", "alpha"], np.array([3.5, 6.0, 10.0, 0.20]),
    np.array([0.0, 0.0, 0.5, 0.0]), np.array([10.0, 10.0, 50.0, 2.0]),
    np.array([5.0, 5.0, 8.0, 0.3]), np.array([3.0, 3.0, 5.0, 0.3]), 0.05, _ac,
    plot_kind="sensors", design=_sens, xlabel="sensor", category="acoustics"))


# ===========================================================================
# --- enzyme kinetics with a competitive inhibitor ---------------------------------------
_S_enz = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
_I_enz = np.array([0.0, 0.5, 1.5, 4.0])
_SS, _II = (a.ravel() for a in np.meshgrid(_S_enz, _I_enz))


def _enz_x(Th, X):
    S, I = X[:, 0], (X[:, 1] if X.shape[1] > 1 else 0.0 * X[:, 0])
    Vm, Km, Ki = Th[:, :1], Th[:, 1:2], Th[:, 2:3]
    return Vm * S / (Km * (1.0 + I / Ki) + S)


def _enz(Th):
    return _enz_x(Th, np.column_stack([_SS, _II]))


register(StaticProblem(
    "enzyme-inhibition", "Michaelis–Menten kinetics with a competitive inhibitor (8 S × 4 I)",
    ["Vmax", "Km", "Ki"], np.array([10.0, 2.0, 0.8]), np.array([0.5, 0.05, 0.01]),
    np.array([50.0, 20.0, 20.0]), np.array([8.0, 3.0, 1.0]), np.array([4.0, 2.0, 1.0]), 0.25,
    _enz, x=_SS, groups=np.array([f"I = {i:g}" for i in _II]), xlabel="substrate S", xlog=True,
    category="biochemistry", inputs=("S", "I"), xfun=_enz_x))

# --- one-compartment oral pharmacokinetics (Bateman function) ----------------------------
_t_pk = np.array([0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10, 12, 16, 24.0])
_dose_pk = 500.0


def _pk_x(Th, X):
    t = X[:, 0]
    ka, ke, V = Th[:, :1], Th[:, 1:2], Th[:, 2:3]
    d = np.where(np.abs(ka - ke) < 1e-9, 1e-9, ka - ke)
    return _dose_pk * ka / (V * d) * (np.exp(-ke * t) - np.exp(-ka * t))


def _pk(Th):
    return _pk_x(Th, _t_pk[:, None])


register(StaticProblem(
    "pharmacokinetics", "Oral one-compartment PK, C(t) = D kₐ/(V(kₐ−kₑ))(e^{−kₑt} − e^{−kₐt}); "
    "kₐ > kₑ excludes the flip-flop mode",
    ["ka", "ke", "V"], np.array([1.2, 0.18, 25.0]), np.array([0.3, 0.01, 1.0]),
    np.array([6.0, 0.29, 200.0]), np.array([1.0, 0.15, 30.0]), np.array([1.0, 0.1, 15.0]), 0.3,
    _pk, x=_t_pk, xlabel="time [h]", category="pharmacology", inputs=("t",), xfun=_pk_x))

# --- overlapping Lorentzian peaks on a sloped baseline -----------------------------------
_x_sp = np.linspace(0.0, 10.0, 200)


def _peaks_x(Th, X):
    x = X[:, 0]
    a1, c1, w1, a2, c2, w2, b0, b1 = (Th[:, i:i + 1] for i in range(8))
    L = lambda a, c, w: a / (1.0 + ((x - c) / w) ** 2)  # noqa: E731
    return L(a1, c1, w1) + L(a2, c2, w2) + b0 + b1 * x


def _peaks(Th):
    return _peaks_x(Th, _x_sp[:, None])


register(StaticProblem(
    "spectral-peaks", "Two overlapping Lorentzian lines + linear baseline (8 parameters)",
    ["a1", "c1", "w1", "a2", "c2", "w2", "b0", "b1"],
    np.array([1.0, 4.2, 0.6, 0.7, 5.3, 0.8, 0.1, 0.02]),
    np.array([0.05, 2.0, 0.05, 0.05, 4.8, 0.05, -1.0, -0.5]),
    np.array([5.0, 4.8, 3.0, 5.0, 8.0, 3.0, 1.0, 0.5]),
    np.array([0.8, 4.0, 0.5, 0.8, 5.5, 0.5, 0.0, 0.0]),
    np.array([0.5, 0.5, 0.3, 0.5, 0.5, 0.3, 0.2, 0.05]), 0.03, _peaks, x=_x_sp,
    xlabel="wavenumber (a.u.)", category="spectroscopy", inputs=("x",), xfun=_peaks_x))

# --- exoplanet radial velocity (Keplerian orbit) -----------------------------------------
_t_rv = np.sort(np.random.default_rng(42).uniform(0.0, 200.0, 45))


def _kepler_E(M, e, n_iter=12):
    E = M + e * np.sin(M)
    for _ in range(n_iter):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return E


def _rv_x(Th, X):
    t = X[:, 0]
    P, K, e, w, M0, g = (Th[:, i:i + 1] for i in range(6))
    E = _kepler_E(2 * np.pi * t / P + M0, e)
    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    return g + K * (np.cos(nu + w) + e * np.cos(w))


def _rv(Th):
    return _rv_x(Th, _t_rv[:, None])


register(StaticProblem(
    "radial-velocity", "Exoplanet radial velocity: Keplerian orbit (Newton–Kepler solve), "
    "45 irregular epochs",
    ["P", "K", "e", "omega", "M0", "gamma"], np.array([42.0, 25.0, 0.3, 1.0, 2.0, 5.0]),
    np.array([30.0, 1.0, 0.0, -np.pi, -np.pi, -50.0]),
    np.array([60.0, 100.0, 0.8, np.pi, np.pi, 50.0]),
    np.array([41.0, 20.0, 0.2, 0.8, 1.8, 0.0]), np.array([3.0, 10.0, 0.2, 1.0, 1.0, 10.0]), 3.0,
    _rv, x=_t_rv, xlabel="time [d]", category="astronomy", inputs=("t",), xfun=_rv_x))

# --- SIR epidemic, daily incidence ----------------------------------------------------------
_days_sir = 120


def _sir_scalar(b, g, l0, sub, n_days=_days_sir):
    I = 10.0 ** l0
    S = 1.0 - I
    h = 1.0 / sub
    out = []
    for _ in range(n_days):
        S0 = S
        for _ in range(sub):
            f1 = b * S * I
            S2, I2 = S - h / 2 * f1, I + h / 2 * (f1 - g * I)
            f2 = b * S2 * I2
            S3, I3 = S - h / 2 * f2, I + h / 2 * (f2 - g * I2)
            f3 = b * S3 * I3
            S4, I4 = S - h * f3, I + h * (f3 - g * I3)
            f4 = b * S4 * I4
            S = S - h / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
            I = I + h / 6 * (f1 - g * I + 2 * (f2 - g * I2) + 2 * (f3 - g * I3) + f4 - g * I4)
        out.append(1e4 * (S0 - S))
    return out


def _sir(Th, sub=1, n_days=_days_sir):
    """SIR with RK4 at one step per day (max error vs. Δt = 1/16: 0.02 cases, noise σ = 4).
    The time loop dominates, so batches up to 32 use a pure-float loop (50 µs per θ vs
    ~2 ms flat for the array path)."""
    if len(Th) <= 32:
        return np.array([_sir_scalar(*th, sub, n_days) for th in Th.tolist()])
    b, g, l0 = Th[:, 0], Th[:, 1], Th[:, 2]
    I = 10.0 ** l0
    S = 1.0 - I
    h = 1.0 / sub
    out = np.empty((len(Th), n_days))
    for d in range(n_days):
        S0 = S
        for _ in range(sub):
            f1 = b * S * I
            S2, I2 = S - h / 2 * f1, I + h / 2 * (f1 - g * I)
            f2 = b * S2 * I2
            S3, I3 = S - h / 2 * f2, I + h / 2 * (f2 - g * I2)
            f3 = b * S3 * I3
            S4, I4 = S - h * f3, I + h * (f3 - g * I3)
            f4 = b * S4 * I4
            S = S - h / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
            I = I + h / 6 * (f1 - g * I + 2 * (f2 - g * I2) + 2 * (f3 - g * I3) + f4 - g * I4)
        out[:, d] = 1e4 * (S0 - S)            # new cases per 10 000 that day
    return out


register(StaticProblem(
    "sir-epidemic", "SIR epidemic: daily incidence per 10⁴ (RK4), unknown β, γ, log₁₀ I₀",
    ["beta", "gamma", "log10_I0"], np.array([0.35, 0.12, -4.0]), np.array([0.05, 0.01, -7.0]),
    np.array([2.0, 1.0, -1.0]), np.array([0.3, 0.1, -4.5]), np.array([0.2, 0.1, 1.0]), 4.0,
    _sir, x=np.arange(1.0, _days_sir + 1), xlabel="day", category="epidemiology"))

# --- Theis pumping test --------------------------------------------------------------------
_r_th = np.array([10.0, 30.0, 80.0])
_t_th = np.geomspace(60.0, 2e5, 20)
_RR, _TT = (a.ravel() for a in np.meshgrid(_r_th, _t_th, indexing="ij"))
_Q_th = 0.01


def _theis_x(Th, X):
    from scipy.special import exp1
    t, r = X[:, 0], (X[:, 1] if X.shape[1] > 1 else np.full(len(X), 10.0))
    T, S = 10.0 ** Th[:, :1], 10.0 ** Th[:, 1:2]
    u = r**2 * S / (4.0 * T * t)
    return _Q_th / (4 * np.pi * T) * exp1(u)


def _theis(Th):
    return _theis_x(Th, np.column_stack([_TT, _RR]))


register(StaticProblem(
    "theis-well", "Theis pumping test: drawdown at 3 observation wells, log₁₀ T and log₁₀ S",
    ["log10_T", "log10_S"], np.array([-2.5, -4.0]), np.array([-6.0, -8.0]),
    np.array([0.0, -1.0]), np.array([-3.0, -3.5]), np.array([1.0, 1.0]), 0.01, _theis,
    x=_TT, groups=np.array([f"r = {r:g} m" for r in _RR]), xlabel="time [s]", xlog=True,
    category="hydrology", inputs=("t", "r"), xfun=_theis_x))


# --- SIR at arbitrary (integer) days --------------------------------------------------------
def _sir_x(Th, X):
    days = np.clip(np.rint(X[:, 0]).astype(int), 1, None)
    return _sir(Th, n_days=int(days.max()))[:, days - 1]


_STATIC["sir-epidemic"] = replace(_STATIC["sir-epidemic"], inputs=("day",), xfun=_sir_x)


# ===========================================================================
# General-purpose curves, mainly for your own data (start values and bounds from the data)
def _span(v):
    return float(np.ptp(v)) or 1.0


def _expdecay_x(Th, X):
    A, k, c = Th[:, :1], Th[:, 1:2], Th[:, 2:3]
    return A * np.exp(-k * X[:, 0]) + c


def _expdecay_guess(X, y):
    x = X[:, 0]
    o = np.argsort(x)
    c = float(np.median(y[o][-max(3, len(y) // 10):]))
    A = float(y[o][0] - c) or _span(y)
    k = 3.0 / _span(x)
    sy = _span(y)
    return ([A, k, c], [-20 * sy - abs(A), k * 1e-3, -20 * sy - abs(c)],
            [20 * sy + abs(A), k * 1e3, 20 * sy + abs(c)])


_t_ed = np.linspace(0.0, 10.0, 60)
register(StaticProblem(
    "exp-decay", "Exponential decay to an offset, y = A e^{−kx} + c (half-lives, cooling, discharge)",
    ["A", "k", "c"], np.array([4.0, 0.45, 0.8]), np.array([-100.0, 1e-3, -100.0]),
    np.array([100.0, 50.0, 100.0]), np.array([3.0, 0.3, 0.5]), np.array([3.0, 0.3, 1.0]), 0.08,
    lambda Th: _expdecay_x(Th, _t_ed[:, None]), x=_t_ed, xlabel="x", category="general",
    inputs=("x",), xfun=_expdecay_x, guess=_expdecay_guess))


def _logistic_x(Th, X):
    K, r, x0, b = (Th[:, i:i + 1] for i in range(4))
    return b + K / (1.0 + np.exp(-r * (X[:, 0] - x0)))


def _logistic_guess(X, y):
    x = X[:, 0]
    o = np.argsort(x)
    b = float(np.min(y)); K = _span(y)
    half = b + K / 2
    x0 = float(x[o][np.argmin(np.abs(y[o] - half))])
    r = 8.0 / _span(x)
    sx, sy = _span(x), _span(y)
    return ([K, r, x0, b], [0.0, r * 1e-3, x.min() - sx, b - 10 * sy],
            [20 * sy, r * 1e3, x.max() + sx, b + 10 * sy])


_t_lg = np.linspace(0.0, 30.0, 50)
register(StaticProblem(
    "logistic", "Logistic growth / dose–response, y = b + K / (1 + e^{−r(x − x₀)})",
    ["K", "r", "x0", "b"], np.array([10.0, 0.5, 12.0, 1.0]), np.array([0.0, 1e-3, -50.0, -50.0]),
    np.array([100.0, 20.0, 80.0, 50.0]), np.array([8.0, 0.4, 14.0, 0.5]),
    np.array([5.0, 0.3, 5.0, 1.0]), 0.3, lambda Th: _logistic_x(Th, _t_lg[:, None]), x=_t_lg,
    xlabel="x", category="general", inputs=("x",), xfun=_logistic_x, guess=_logistic_guess))


def _gauss_x(Th, X):
    a, mu, s, b0, b1 = (Th[:, i:i + 1] for i in range(5))
    x = X[:, 0]
    return a * np.exp(-0.5 * ((x - mu) / s) ** 2) + b0 + b1 * x


def _gauss_guess(X, y):
    x = X[:, 0]
    o = np.argsort(x)
    xs_, ys_ = x[o], y[o]
    b0 = float(np.median(np.r_[ys_[:max(2, len(y) // 10)], ys_[-max(2, len(y) // 10):]]))
    j = int(np.argmax(np.abs(ys_ - b0)))
    a = float(ys_[j] - b0); mu = float(xs_[j])
    above = np.abs(ys_ - b0) > abs(a) / 2
    s = max(float(np.ptp(xs_[above])) / 2.355, _span(x) / 200) if above.sum() > 1 else _span(x) / 10
    sx, sy = _span(x), _span(y)
    return ([a, mu, s, b0, 0.0], [-20 * sy, x.min(), sx / 1e3, b0 - 10 * sy, -10 * sy / sx],
            [20 * sy, x.max(), sx, b0 + 10 * sy, 10 * sy / sx])


_t_gp = np.linspace(-5.0, 5.0, 80)
register(StaticProblem(
    "gaussian-peak", "Gaussian peak on a linear baseline (chromatography, line profiles)",
    ["a", "mu", "s", "b0", "b1"], np.array([2.0, 0.7, 0.9, 0.3, 0.05]),
    np.array([-50.0, -5.0, 0.01, -10.0, -5.0]), np.array([50.0, 5.0, 10.0, 10.0, 5.0]),
    np.array([1.5, 0.0, 1.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.5, 0.5, 0.1]), 0.06,
    lambda Th: _gauss_x(Th, _t_gp[:, None]), x=_t_gp, xlabel="x", category="general",
    inputs=("x",), xfun=_gauss_x, guess=_gauss_guess))
