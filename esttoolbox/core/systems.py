"""Benchmark state-space systems.

Each entry is an `SSMProblem`: a parametric family theta -> StateSpaceModel,
the true parameters, bounds and a prior (used by system identification), the
default horizon, and plotting hints.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .linalg import van_loan
from .ssm import LinearGaussianSSM, StateSpaceModel


@dataclass
class SSMProblem:
    name: str
    description: str
    make: callable = field(repr=False)          # theta -> StateSpaceModel
    param_names: list
    theta_true: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    T: int
    prior_mean: np.ndarray = None               # prior centre (natural units)
    prior_std: np.ndarray = None                # prior std on the unconstrained scale (sysid.Transform)
    xy: tuple | None = None                     # state indices for a 2-D track plot
    recommended: tuple = ()
    meas_xy: callable = field(default=None, repr=False)  # ys -> (T, 2) positions for track overlays
    landmarks: dict = field(default_factory=dict)  # name -> (n, 2) fixed points for track plots
    category: str = "nonlinear"                 # linear / nonlinear / chaotic / non-Gaussian / engineering

    def model(self, theta=None):
        return self.make(self.theta_true if theta is None else np.asarray(theta, float))


_SSM: dict[str, SSMProblem] = {}


def register(p):
    _SSM[p.name] = p
    return p


def get_ssm(name):
    try:
        return _SSM[name]
    except KeyError:
        raise KeyError(f"unknown state-space system '{name}'; available: {', '.join(_SSM)}")


def list_ssm():
    return dict(_SSM)


# --------------------------------------------------------------------------
# 1. constant-velocity tracking (linear Gaussian)
# --------------------------------------------------------------------------
def _cv(theta):
    q, r = theta
    dt = 1.0
    A1 = np.array([[0.0, 1.0], [0.0, 0.0]])
    F1, Q1 = van_loan(A1, np.array([[0.0], [1.0]]), np.array([[q]]), dt)
    # state [px, py, vx, vy]
    A = np.zeros((4, 4)); Q = np.zeros((4, 4))
    for i, (p, v) in enumerate([(0, 2), (1, 3)]):
        A[np.ix_([p, v], [p, v])] = F1
        Q[np.ix_([p, v], [p, v])] = Q1
    C = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    return LinearGaussianSSM(
        "cv-tracking", A, C, Q, r**2 * np.eye(2),
        m0=[0, 0, 1, 0.5], P0=np.diag([4.0, 4.0, 1.0, 1.0]), x0=[0, 0, 1, 0.5],
        state_names=["px", "py", "vx", "vy"], obs_names=["px", "py"], dt=dt,
        description="2-D white-noise-acceleration target, noisy positions (linear Gaussian)")


register(SSMProblem(
    "cv-tracking", "2-D constant-velocity target, position measurements (linear Gaussian)",
    _cv, ["q", "r"], np.array([0.05, 1.0]), np.array([1e-4, 1e-2]), np.array([10.0, 20.0]),
    T=100, prior_mean=np.array([0.1, 1.0]), prior_std=np.array([1.5, 1.5]), xy=(0, 1),
    recommended=("kf", "srkf", "ukf", "pf"), meas_xy=lambda ys: ys, category="linear"))


# --------------------------------------------------------------------------
# 2. pendulum, y = sin(angle)   (Särkkä 2013, Ex. 5.1)
# --------------------------------------------------------------------------
def _pend(theta):
    g, qc, r = theta
    dt = 0.01

    def f(X, k):
        a, w = X[..., 0], X[..., 1]
        return np.stack([a + w * dt, w - g * np.sin(a) * dt], axis=-1)

    def Fj(x, k):
        return np.array([[1.0, dt], [-g * np.cos(x[0]) * dt, 1.0]])

    Q = qc * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    return StateSpaceModel(
        "pendulum", 2, 1, f, lambda X, k: np.sin(X[..., :1]), Q, [[r**2]],
        m0=[1.6, 0.0], P0=np.diag([0.1, 0.1]), x0=[1.5, 0.0],
        F_jac=Fj, H_jac=lambda x, k: np.array([[np.cos(x[0]), 0.0]]),
        state_names=["angle", "rate"], obs_names=["sin(angle)"], dt=dt,
        description="Nonlinear pendulum, horizontal-position measurement sin(α)")


register(SSMProblem(
    "pendulum", "Nonlinear pendulum, measurement sin(angle) (Särkkä 2013)",
    _pend, ["g", "q_c", "r"], np.array([9.81, 0.01, 0.3]),
    np.array([1.0, 1e-5, 1e-2]), np.array([30.0, 1.0, 3.0]), T=500,
    prior_mean=np.array([8.0, 0.02, 0.5]), prior_std=np.array([0.5, 1.5, 1.0]),
    recommended=("ekf", "ukf", "ckf", "pf")))


# --------------------------------------------------------------------------
# 3. coordinated turn with unknown turn rate, range/bearing radar
# --------------------------------------------------------------------------
def _ct_f(dt):
    def f(X, k):
        px, vx, py, vy, w = (X[..., i] for i in range(5))
        wt = w * dt
        small = np.abs(wt) < 1e-6
        ws = np.where(small, 1.0, w)
        s, c = np.sin(wt), np.cos(wt)
        a = np.where(small, dt - w**2 * dt**3 / 6, s / ws)          # sin(wT)/w
        b = np.where(small, w * dt**2 / 2, (1 - c) / ws)            # (1-cos wT)/w
        return np.stack([px + a * vx - b * vy, c * vx - s * vy,
                         py + b * vx + a * vy, s * vx + c * vy, w], axis=-1)
    return f


def _ct(theta):
    sa, sw, sr, sb = theta
    dt = 1.0
    G = np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]) * sa**2
    Q = np.zeros((5, 5))
    Q[np.ix_([0, 1], [0, 1])] = G
    Q[np.ix_([2, 3], [2, 3])] = G
    Q[4, 4] = (sw * dt) ** 2

    def h(X, k):
        px, py = X[..., 0], X[..., 2]
        return np.stack([np.hypot(px, py), np.arctan2(py, px)], axis=-1)

    def Hj(x, k):
        px, py = x[0], x[2]
        r2 = px * px + py * py
        r = np.sqrt(r2)
        return np.array([[px / r, 0, py / r, 0, 0], [-py / r2, 0, px / r2, 0, 0]])

    x0 = np.array([400.0, -8.0, 300.0, 6.0, np.deg2rad(3.0)])
    return StateSpaceModel(
        "ct-radar", 5, 2, _ct_f(dt), h, Q, np.diag([sr**2, sb**2]),
        m0=x0 + np.array([20.0, 2.0, -20.0, -2.0, 0.0]),
        P0=np.diag([30.0**2, 3.0**2, 30.0**2, 3.0**2, np.deg2rad(3.0) ** 2]), x0=x0,
        H_jac=Hj, state_names=["px", "vx", "py", "vy", "omega"],
        obs_names=["range", "bearing"], dt=dt, angle_obs=(1,),
        description="Coordinated turn with unknown turn rate, range/bearing radar at origin")


register(SSMProblem(
    "ct-radar", "Coordinated-turn target with unknown turn rate, range/bearing radar",
    _ct, ["sigma_a", "sigma_omega", "sigma_r", "sigma_b"],
    np.array([0.2, np.deg2rad(0.2), 5.0, np.deg2rad(2.0)]),
    np.array([1e-3, 1e-5, 0.1, 1e-4]), np.array([10.0, 0.2, 100.0, 0.5]), T=120,
    prior_mean=np.array([0.3, np.deg2rad(0.3), 5.0, np.deg2rad(2.0)]),
    prior_std=np.array([1.0, 1.0, 0.5, 0.5]), xy=(0, 2),
    recommended=("ekf", "ukf", "ckf", "iplf", "pf"), category="tracking",
    meas_xy=lambda ys: np.column_stack([ys[:, 0] * np.cos(ys[:, 1]), ys[:, 0] * np.sin(ys[:, 1])]),
    landmarks={"radar": np.zeros((1, 2))}))


# --------------------------------------------------------------------------
# 4. univariate nonstationary growth model (Kitagawa / Gordon et al. 1993)
# --------------------------------------------------------------------------
def _ungm(theta):
    q, r = theta

    def f(X, k):
        x = X[..., :1]
        return 0.5 * x + 25 * x / (1 + x * x) + 8 * np.cos(1.2 * k)

    def Fj(x, k):
        return np.array([[0.5 + 25 * (1 - x[0] ** 2) / (1 + x[0] ** 2) ** 2]])

    return StateSpaceModel(
        "ungm", 1, 1, f, lambda X, k: X[..., :1] ** 2 / 20, [[q]], [[r]],
        m0=[0.0], P0=[[5.0]], x0=None, F_jac=Fj, H_jac=lambda x, k: np.array([[x[0] / 10]]),
        state_names=["x"], obs_names=["x²/20"],
        description="Univariate nonstationary growth model: bimodal posterior (sign ambiguity)")


register(SSMProblem(
    "ungm", "Univariate nonstationary growth model, y = x²/20 (bimodal posterior)",
    _ungm, ["q", "r"], np.array([10.0, 1.0]), np.array([0.1, 0.01]), np.array([100.0, 20.0]),
    T=100, prior_mean=np.array([5.0, 1.0]), prior_std=np.array([1.0, 1.0]),
    recommended=("ekf", "ukf", "pf", "pf-ekf"), category="multimodal"))


# --------------------------------------------------------------------------
# 5. Lorenz-63 with RK4 between observations
# --------------------------------------------------------------------------
def _lorenz(theta):
    s, rho, beta, q = theta
    dt, nsub = 0.1, 4
    h = dt / nsub

    def rhs(X):
        x, y, z = X[..., 0], X[..., 1], X[..., 2]
        return np.stack([s * (y - x), x * (rho - z) - y, x * y - beta * z], axis=-1)

    def f(X, k):
        for _ in range(nsub):
            k1 = rhs(X); k2 = rhs(X + h / 2 * k1); k3 = rhs(X + h / 2 * k2); k4 = rhs(X + h * k3)
            X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return X

    return StateSpaceModel(
        "lorenz63", 3, 3, f, lambda X, k: X.copy(), q * np.eye(3), 4.0 * np.eye(3),
        m0=[1.0, 1.0, 25.0], P0=25.0 * np.eye(3), x0=[-5.9, -5.5, 24.6],
        state_names=["x", "y", "z"], obs_names=["x", "y", "z"], dt=dt,
        description="Chaotic Lorenz-63, RK4 over Δt = 0.1, all components observed (R = 4I)")


register(SSMProblem(
    "lorenz63", "Chaotic Lorenz-63 (RK4, Δt = 0.1), noisy full-state observations",
    _lorenz, ["sigma", "rho", "beta", "q"], np.array([10.0, 28.0, 8.0 / 3.0, 0.1]),
    np.array([1.0, 5.0, 0.5, 1e-4]), np.array([30.0, 60.0, 10.0, 10.0]), T=200,
    prior_mean=np.array([8.0, 25.0, 3.0, 0.2]), prior_std=np.array([0.3, 0.3, 0.3, 1.5]),
    recommended=("ekf", "ukf", "enkf", "etkf", "pf"), category="chaotic"))


# --------------------------------------------------------------------------
# 6. stochastic volatility  (non-additive measurement: PF / PMMH only)
# --------------------------------------------------------------------------
class StochasticVolatility(StateSpaceModel):
    gaussian_obs = False

    def __init__(self, phi, sigma, beta):
        self.phi, self.sigma, self.beta = phi, sigma, beta
        P0 = sigma**2 / max(1 - phi**2, 1e-6)
        super().__init__("stochastic-volatility", 1, 1, lambda X, k: phi * X,
                         lambda X, k: np.zeros(X.shape[:-1] + (1,)), [[sigma**2]], [[1.0]],
                         m0=[0.0], P0=[[P0]], F_jac=lambda x, k: np.array([[phi]]),
                         state_names=["log-vol"], obs_names=["return"],
                         description="y_k = β exp(x_k/2) ε_k, x_k = φ x_{k-1} + σ η_k")

    def obs_loglik(self, y, X, k):
        v = self.beta**2 * np.exp(X[:, 0])
        return -0.5 * (np.log(2 * np.pi * v) + y[0] ** 2 / v)

    def sample_obs(self, X, k, rng):
        return self.beta * np.exp(X[..., :1] / 2) * rng.standard_normal(X[..., :1].shape)


register(SSMProblem(
    "stochastic-volatility", "Stochastic volatility, y = β exp(x/2) ε (non-additive; PF / PMMH)",
    lambda th: StochasticVolatility(*th), ["phi", "sigma", "beta"],
    np.array([0.91, 1.0, 0.5]), np.array([-0.999, 0.01, 0.01]), np.array([0.999, 5.0, 5.0]),
    T=200, prior_mean=np.array([0.8, 0.7, 0.7]), prior_std=np.array([1.0, 1.0, 1.0]),
    recommended=("pf", "pf-ekf"), category="non-Gaussian"))


# --------------------------------------------------------------------------
# 7. gas-phase reactor 2A -> B  (Haseltine & Rawlings 2005): EKF vs constrained MHE
# --------------------------------------------------------------------------
def _reactor(theta):
    kr, r = theta
    dt, nsub = 0.25, 5
    h = dt / nsub

    def rhs(X):
        a = X[..., 0]
        return np.stack([-2 * kr * a * a, kr * a * a], axis=-1)

    def f(X, k):
        for _ in range(nsub):
            k1 = rhs(X); k2 = rhs(X + h / 2 * k1); k3 = rhs(X + h / 2 * k2); k4 = rhs(X + h * k3)
            X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return X

    return StateSpaceModel(
        "reactor", 2, 1, f, lambda X, k: X[..., :1] + X[..., 1:2], 1e-6 * np.eye(2), [[r**2]],
        m0=[0.1, 4.5], P0=36.0 * np.eye(2), x0=[3.0, 1.0],
        H_jac=lambda x, k: np.array([[1.0, 1.0]]), state_names=["pA", "pB"],
        obs_names=["pA + pB"], dt=dt, state_bounds=(np.zeros(2), np.full(2, np.inf)),
        description="Gas-phase reactor 2A→B, total-pressure measurement, poor prior; "
                    "unconstrained EKF converges to negative pressures")


register(SSMProblem(
    "reactor", "Gas-phase reactor 2A → B with a poor prior (EKF vs constrained MHE)",
    _reactor, ["k", "r"], np.array([0.16, 0.1]), np.array([0.01, 0.01]), np.array([2.0, 2.0]),
    T=120, prior_mean=np.array([0.1, 0.2]), prior_std=np.array([1.0, 1.0]),
    recommended=("ekf", "ckf", "mhe", "rpf"), category="constrained"))


# ==========================================================================
# helpers for time-varying inputs: k may be a scalar or an array broadcast
# against X[..., :1]; these return values shaped like X[..., 0]
# ==========================================================================
def _at(table, k):
    kk = np.asarray(k)
    v = table[np.clip(np.rint(kk).astype(int), 0, len(table) - 1)]
    return v[..., 0] if kk.ndim else v


def _rk4(rhs, X, h, n):
    for _ in range(n):
        k1 = rhs(X); k2 = rhs(X + h / 2 * k1); k3 = rhs(X + h / 2 * k2); k4 = rhs(X + h * k3)
        X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return X


# --------------------------------------------------------------------------
# 8. bearings-only target motion analysis with an ownship manoeuvre
# --------------------------------------------------------------------------
_BO_T = 400
_bo_k = np.arange(_BO_T + 1, dtype=float)
_bo_head = np.where(_bo_k < 30, 0.0, np.where(_bo_k < 60, np.deg2rad(120.0), np.deg2rad(30.0)))
_bo_own = np.concatenate([[[0.0, 0.0]], np.cumsum(8.0 * np.column_stack(
    [np.cos(_bo_head[1:]), np.sin(_bo_head[1:])]), axis=0)])     # ownship track, 8 per step


def _bo(theta):
    q, sb = theta
    F1, Q1 = van_loan(np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([[0.0], [1.0]]),
                      np.array([[q]]), 1.0)
    A = np.kron(np.eye(2), F1)[np.ix_([0, 2, 1, 3], [0, 2, 1, 3])]      # [px, py, vx, vy]
    Q = np.kron(np.eye(2), Q1)[np.ix_([0, 2, 1, 3], [0, 2, 1, 3])]

    def h(X, k):
        dx = X[..., 0] - _at(_bo_own[:, 0], k)
        dy = X[..., 1] - _at(_bo_own[:, 1], k)
        return np.arctan2(dy, dx)[..., None]

    def Hj(x, k):
        dx, dy = x[0] - _at(_bo_own[:, 0], k), x[1] - _at(_bo_own[:, 1], k)
        r2 = dx * dx + dy * dy
        return np.array([[-dy / r2, dx / r2, 0.0, 0.0]])

    x0 = np.array([800.0, 1500.0, -4.0, -2.0])
    return StateSpaceModel(
        "bearings-only", 4, 1, lambda X, k: X @ A.T, h, Q, [[sb**2]],
        m0=[1200.0, 1200.0, 0.0, 0.0], P0=np.diag([500.0**2, 500.0**2, 4.0**2, 4.0**2]), x0=x0,
        F_jac=lambda x, k: A, H_jac=Hj, state_names=["px", "py", "vx", "vy"],
        obs_names=["bearing"], angle_obs=(0,),
        description="Bearings-only target motion analysis: range is unobservable until the "
                    "ownship turns (legs change at k = 30, 60)")


register(SSMProblem(
    "bearings-only", "Bearings-only tracking (TMA): ownship manoeuvres make range observable",
    _bo, ["q", "sigma_b"], np.array([1e-3, np.deg2rad(1.0)]), np.array([1e-6, 1e-4]),
    np.array([1.0, 0.2]), T=200, prior_mean=np.array([2e-3, np.deg2rad(1.5)]),
    prior_std=np.array([1.5, 0.7]), xy=(0, 1), recommended=("ekf", "ukf", "iplf", "rpf"),
    category="tracking", landmarks={"ownship": _bo_own[:201]}))


# --------------------------------------------------------------------------
# 9. mobile-robot localization: unicycle odometry + range/bearing to landmarks
# --------------------------------------------------------------------------
_RB_T = 2000
_rb_dt = 0.5
_rb_k = np.arange(_RB_T + 1)
_rb_v = 1.0 + 0.3 * np.sin(2 * np.pi * _rb_k / 150)
_rb_w = 0.35 * np.sin(2 * np.pi * _rb_k / 120)
_rb_lm = np.array([[5.0, 12.0], [15.0, -10.0], [25.0, 11.0], [35.0, -8.0], [45.0, 9.0]])


def _robot(theta):
    sv, sw, sr, sb = theta
    dt, L = _rb_dt, _rb_lm
    nl = len(L)

    def f(X, k):
        v, w = _at(_rb_v, k), _at(_rb_w, k)
        th = X[..., 2]
        return np.stack([X[..., 0] + v * dt * np.cos(th), X[..., 1] + v * dt * np.sin(th),
                         th + w * dt], axis=-1)

    def Fj(x, k):
        v = _at(_rb_v, k)
        return np.array([[1, 0, -v * dt * np.sin(x[2])], [0, 1, v * dt * np.cos(x[2])], [0, 0, 1]])

    def h(X, k):
        dx = L[:, 0] - X[..., 0, None]
        dy = L[:, 1] - X[..., 1, None]
        b = np.arctan2(dy, dx) - X[..., 2, None]
        b = (b + np.pi) % (2 * np.pi) - np.pi
        return np.stack([np.hypot(dx, dy), b], axis=-1).reshape(X.shape[:-1] + (2 * nl,))

    def Hj(x, k):
        H = np.zeros((2 * nl, 3))
        for i, (lx, ly) in enumerate(L):
            dx, dy = lx - x[0], ly - x[1]
            r2 = dx * dx + dy * dy
            r = np.sqrt(r2)
            H[2 * i] = [-dx / r, -dy / r, 0.0]
            H[2 * i + 1] = [dy / r2, -dx / r2, -1.0]
        return H

    Q = np.diag([(sv * dt) ** 2, (sv * dt) ** 2, (sw * dt) ** 2])
    R = np.diag(np.tile([sr**2, sb**2], nl))
    names = [n for i in range(nl) for n in (f"r{i + 1}", f"b{i + 1}")]
    return StateSpaceModel(
        "robot-landmarks", 3, 2 * nl, f, h, Q, R, m0=[1.0, -1.0, 0.2], P0=np.diag([1.0, 1.0, 0.1]),
        x0=[0.0, 0.0, 0.0], F_jac=Fj, H_jac=Hj, state_names=["x", "y", "heading"],
        obs_names=names, dt=dt, angle_obs=tuple(range(1, 2 * nl, 2)),
        description="Unicycle robot with known odometry inputs, range and bearing to five "
                    "known landmarks (EKF-SLAM localization step)")


register(SSMProblem(
    "robot-landmarks", "Mobile-robot localization: odometry + range/bearing to 5 landmarks",
    _robot, ["sigma_v", "sigma_w", "sigma_r", "sigma_b"], np.array([0.1, 0.05, 0.5, 0.05]),
    np.array([1e-3, 1e-4, 1e-2, 1e-4]), np.array([5.0, 2.0, 10.0, 1.0]), T=200,
    prior_mean=np.array([0.15, 0.08, 0.7, 0.07]), prior_std=np.array([1.0, 1.0, 1.0, 1.0]),
    xy=(0, 1), recommended=("ekf", "iekf", "ukf", "pf"), category="robotics",
    landmarks={"landmarks": _rb_lm}))


# --------------------------------------------------------------------------
# 10. Lorenz-96 (40 variables), every other component observed: ensemble methods
# --------------------------------------------------------------------------
def _l96_rhs(F):
    def rhs(X):
        return (np.roll(X, -1, -1) - np.roll(X, 2, -1)) * np.roll(X, 1, -1) - X + F
    return rhs


def _l96_attractor(F=8.0, n=40):
    x = F + 0.01 * np.sin(np.arange(n))
    return _rk4(_l96_rhs(F), x, 0.05, 1000)


_l96_x = _l96_attractor()


def _l96(theta):
    Fc, q = theta
    n, dt = 40, 0.05
    rhs = _l96_rhs(Fc)
    C = np.eye(n)[::2]
    return StateSpaceModel(
        "lorenz96", n, n // 2, lambda X, k: _rk4(rhs, X, dt, 1), lambda X, k: X[..., ::2].copy(),
        q * np.eye(n), np.eye(n // 2), m0=_l96_x, P0=np.eye(n),
        H_jac=lambda x, k: C, state_names=[f"x{i + 1}" for i in range(n)],
        obs_names=[f"x{i + 1}" for i in range(0, n, 2)], dt=dt,
        description="Lorenz-96, 40 variables, forcing F, RK4 Δt = 0.05, odd components observed "
                    "(R = I): a data-assimilation benchmark for ensemble filters")


register(SSMProblem(
    "lorenz96", "Lorenz-96 (40-D, F = 8), half the variables observed (ensemble DA)",
    _l96, ["F", "q"], np.array([8.0, 0.01]), np.array([2.0, 1e-5]), np.array([20.0, 1.0]),
    T=200, prior_mean=np.array([7.0, 0.02]), prior_std=np.array([0.3, 1.5]),
    recommended=("ekf", "ukf", "enkf", "etkf"), category="chaotic"))


# --------------------------------------------------------------------------
# 11. Van der Pol oscillator, position only
# --------------------------------------------------------------------------
def _vdp(theta):
    mu, q, r = theta
    dt, nsub = 0.1, 4

    def rhs(X):
        x, v = X[..., 0], X[..., 1]
        return np.stack([v, mu * (1 - x * x) * v - x], axis=-1)

    Q = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    return StateSpaceModel(
        "vanderpol", 2, 1, lambda X, k: _rk4(rhs, X, dt / nsub, nsub), lambda X, k: X[..., :1].copy(),
        Q, [[r**2]], m0=[0.0, 0.0], P0=np.diag([4.0, 4.0]), x0=[2.0, 0.0],
        H_jac=lambda x, k: np.array([[1.0, 0.0]]), state_names=["x", "v"], obs_names=["x"],
        dt=dt, description="Van der Pol limit cycle (relaxation oscillation for large μ), "
                           "position measured, velocity hidden")


register(SSMProblem(
    "vanderpol", "Van der Pol oscillator (μ = 2), position measured, velocity hidden",
    _vdp, ["mu", "q", "r"], np.array([2.0, 0.05, 0.2]), np.array([0.05, 1e-5, 1e-3]),
    np.array([10.0, 5.0, 5.0]), T=200, prior_mean=np.array([1.5, 0.1, 0.3]),
    prior_std=np.array([0.5, 1.5, 1.0]), xy=(0, 1), recommended=("ekf", "ukf", "ckf", "pf")))


# --------------------------------------------------------------------------
# 12. stochastic Lotka–Volterra with Poisson counts (non-Gaussian measurement)
# --------------------------------------------------------------------------
class PredatorPrey(StateSpaceModel):
    """Log-populations u = log N (prey), v = log P (predator):
    du = (α − β e^v) dt + σ dW,  dv = (δ e^u − γ) dt + σ dW;  counts y ~ Poisson(e^x)."""
    gaussian_obs = False
    _beta, _delta = 0.028, 0.026

    def __init__(self, alpha, gamma, sigma):
        dt, nsub = 0.5, 5
        b, d = self._beta, self._delta

        def rhs(X):
            u, v = X[..., 0], X[..., 1]
            return np.stack([alpha - b * np.exp(np.minimum(v, 30)),
                             d * np.exp(np.minimum(u, 30)) - gamma], axis=-1)

        x0 = np.log([30.0, 4.0])
        super().__init__("predator-prey", 2, 2, lambda X, k: _rk4(rhs, X, dt / nsub, nsub),
                         lambda X, k: np.exp(np.minimum(X, 30)), sigma**2 * dt * np.eye(2),
                         np.eye(2), m0=x0 + [0.3, -0.3], P0=0.25 * np.eye(2), x0=x0,
                         state_names=["log prey", "log predator"], obs_names=["prey count",
                                                                              "predator count"],
                         dt=dt, description="Stochastic Lotka–Volterra on log-populations, "
                                            "Poisson counts of both species")

    def obs_loglik(self, y, X, k):
        lam = np.exp(np.minimum(X, 30))
        from scipy.special import gammaln
        return np.sum(y * X - lam - gammaln(y + 1), axis=-1)

    def sample_obs(self, X, k, rng):
        return rng.poisson(np.exp(np.minimum(X, 30))).astype(float)


register(SSMProblem(
    "predator-prey", "Stochastic Lotka–Volterra, Poisson counts (non-Gaussian; PF / PMMH)",
    lambda th: PredatorPrey(*th), ["alpha", "gamma", "sigma"], np.array([0.55, 0.84, 0.1]),
    np.array([0.05, 0.05, 0.005]), np.array([3.0, 3.0, 1.0]), T=100,
    prior_mean=np.array([0.5, 0.8, 0.15]), prior_std=np.array([0.5, 0.5, 1.0]), xy=(0, 1),
    recommended=("pf", "rpf"), category="non-Gaussian"))


# --------------------------------------------------------------------------
# 13. Li-ion battery state of charge: Thevenin 1-RC model, pulsed discharge
# --------------------------------------------------------------------------
_BT_T = 3000
_bt_dt = 10.0
_bt_cap = 2.0 * 3600.0                                   # 2 Ah in coulombs
_bt_k = np.arange(_BT_T + 1)
_bt_i = np.where((_bt_k // 30) % 2 == 0, 2.0, 0.0) + np.where((_bt_k // 7) % 5 == 0, 1.0, 0.0)


def _ocv(s):
    return 3.0 + 1.6 * s - 1.5 * s**2 + 1.1 * s**3


def _docv(s):
    return 1.6 - 3.0 * s + 3.3 * s**2


def _battery(theta):
    R0, R1, tau, r = theta
    a = np.exp(-_bt_dt / tau)

    def f(X, k):
        i = _at(_bt_i, k)
        return np.stack([X[..., 0] - _bt_dt * i / _bt_cap, a * X[..., 1] + R1 * (1 - a) * i],
                        axis=-1)

    def h(X, k):
        return (_ocv(X[..., 0]) - X[..., 1] - R0 * _at(_bt_i, k))[..., None]

    return StateSpaceModel(
        "battery-soc", 2, 1, f, h, np.diag([1e-8, 1e-6]), [[r**2]],
        m0=[0.6, 0.0], P0=np.diag([0.25**2, 0.02**2]), x0=[0.95, 0.0],
        F_jac=lambda x, k: np.diag([1.0, a]),
        H_jac=lambda x, k: np.array([[_docv(x[0]), -1.0]]), state_names=["SoC", "V_RC"],
        obs_names=["terminal V"], dt=_bt_dt,
        state_bounds=(np.array([0.0, -np.inf]), np.array([1.0, np.inf])),
        description="Thevenin 1-RC Li-ion cell, pulsed discharge current, terminal-voltage "
                    "measurement, wrong initial SoC (0.6 vs 0.95)")


register(SSMProblem(
    "battery-soc", "Li-ion state of charge (1-RC Thevenin model), wrong initial SoC",
    _battery, ["R0", "R1", "tau", "r"], np.array([0.05, 0.03, 60.0, 0.01]),
    np.array([1e-3, 1e-3, 1.0, 1e-4]), np.array([1.0, 1.0, 2000.0, 0.5]), T=400,
    prior_mean=np.array([0.04, 0.04, 80.0, 0.02]), prior_std=np.array([0.5, 0.7, 0.7, 1.0]),
    recommended=("ekf", "ukf", "mhe", "pf"), category="engineering"))


# --------------------------------------------------------------------------
# 14. ballistic re-entry (Julier & Uhlmann 2004): unknown drag coefficient
# --------------------------------------------------------------------------
_RE = dict(b0=-0.59783, H0=13.406, Gm0=3.9860e5, R0=6374.0, xr=6374.0, yr=0.0)


def _reentry(theta):
    qv, sr, sb = theta
    c = _RE
    dt, nsub = 0.1, 2

    def rhs(X):
        x1, x2, x3, x4, x5 = (X[..., i] for i in range(5))
        R = np.hypot(x1, x2)
        V = np.hypot(x3, x4)
        D = c["b0"] * np.exp(x5) * np.exp((c["R0"] - R) / c["H0"]) * V
        G = -c["Gm0"] / R**3
        z = np.zeros_like(x1)
        return np.stack([x3, x4, D * x3 + G * x1, D * x4 + G * x2, z], axis=-1)

    def h(X, k):
        dx, dy = X[..., 0] - c["xr"], X[..., 1] - c["yr"]
        return np.stack([np.hypot(dx, dy), np.arctan2(dy, dx)], axis=-1)

    def Hj(x, k):
        dx, dy = x[0] - c["xr"], x[1] - c["yr"]
        r2 = dx * dx + dy * dy
        r = np.sqrt(r2)
        return np.array([[dx / r, dy / r, 0, 0, 0], [-dy / r2, dx / r2, 0, 0, 0]])

    x0 = np.array([6500.4, 349.14, -1.8093, -6.7967, 0.6932])
    return StateSpaceModel(
        "reentry", 5, 2, lambda X, k: _rk4(rhs, X, dt / nsub, nsub), h,
        np.diag([1e-12, 1e-12, qv**2, qv**2, 1e-8]), np.diag([sr**2, sb**2]),
        m0=x0 * np.array([1, 1, 1, 1, 0.0]), P0=np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1.0]), x0=x0,
        H_jac=Hj, state_names=["x [km]", "y [km]", "vx [km/s]", "vy [km/s]", "log drag"],
        obs_names=["range [km]", "bearing"], dt=dt, angle_obs=(1,),
        description="Ballistic re-entry vehicle with unknown aerodynamic coefficient, "
                    "range/bearing radar on the surface (Julier & Uhlmann 2004)")


register(SSMProblem(
    "reentry", "Ballistic re-entry with unknown drag (Julier & Uhlmann), radar tracking",
    _reentry, ["q_v", "sigma_r", "sigma_b"], np.array([np.sqrt(2.4064e-5), 1e-3, 1.7e-4]),
    np.array([1e-5, 1e-5, 1e-6]), np.array([0.1, 0.1, 0.01]), T=600,
    prior_mean=np.array([0.004, 1.5e-3, 2e-4]), prior_std=np.array([1.0, 1.0, 1.0]), xy=(0, 1),
    recommended=("ekf", "ukf", "ckf", "iplf"), category="tracking",
    landmarks={"radar": np.array([[_RE["xr"], _RE["yr"]]])}))
