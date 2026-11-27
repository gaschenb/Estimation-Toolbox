"""Process-pool back end for the GUI.

Every estimator call is a task in a pool of worker processes, so methods run in parallel on
separate cores and never compete with the drawing thread for the interpreter lock. Tasks are
plain functions of picklable arguments (problem names, θ, data); models are rebuilt in the
worker. A shared generation counter lets queued tasks of a superseded request return
immediately, and Cancel terminates the pool outright (it is re-created in the background).
Falls back to a thread pool if processes cannot be started."""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from multiprocessing.pool import ThreadPool

import atexit

import numpy as np
from PyQt5 import QtCore

_GEN = None          # shared generation counter inside workers


def _init(gen):
    global _GEN
    _GEN = gen
    try:                      # keep worker processes below the GUI in scheduling priority
        if hasattr(os, "nice"):
            os.nice(5)
    except OSError:
        pass
    import esttoolbox.core.run  # noqa: F401  (warm imports)
    import esttoolbox.core.paramest  # noqa: F401


def _stale(token):
    ch, gen = token
    return _GEN is not None and gen < _GEN[ch]


def _warm():
    return os.getpid()


# ----------------------------------------------------------------------------- tasks
def state_task(token, problem, theta, ys, method, opts, seed, do_smooth):
    if _stale(token):
        return None
    from ..core.mhe import batch_map
    from ..core.run import STATE_METHODS, run_method, smooth
    from ..core.userdata import resolve_problem
    model = resolve_problem(problem).model(theta)
    idx = list(STATE_METHODS).index(method) if method in STATE_METHODS else 99
    rng = np.random.default_rng([seed + 1, idx])
    t0 = time.perf_counter()
    if method == "batch-map":
        return [("batch-map", batch_map(model, ys), time.perf_counter() - t0)]
    r = run_method(method, model, ys, rng, **opts)
    out = [(method, r, time.perf_counter() - t0)]
    if do_smooth and not getattr(r, "diverged", False):
        try:
            s = smooth(r, model, rng)
        except Exception:      # degenerate particle sets etc.
            s = None
        if s is not None:
            out.append((s.method, s, s.runtime))
    return out


def mc_task(token, problem, methods, n_runs, run_ids, T, seed, theta, smoother, opts):
    if _stale(token):
        return None
    from ..core.run import monte_carlo
    from ..core.systems import get_ssm
    return monte_carlo(get_ssm(problem), methods, n_runs, T, seed, theta=theta, smoother=smoother,
                       run_ids=run_ids, **opts)


def static_task(token, problem, theta, y, sigma, method, kw, seed):
    if _stale(token):
        return None
    from dataclasses import replace
    from ..core.paramest import METHODS
    from ..core.static import get_static
    if isinstance(problem, tuple):           # ("data", name, StaticData, θ₀): your own data
        _, name, data, th0 = problem
        prob = get_static(name).with_data(data, th0)[0]
    else:
        prob = replace(get_static(problem), theta_true=np.asarray(theta, float))
    rng = np.random.default_rng([seed + 1, list(METHODS).index(method)])
    t0 = time.perf_counter()
    r = METHODS[method](prob, y, sigma, rng=rng, **kw)
    return [(method, r, time.perf_counter() - t0)]


# ----------------------------------------------------------------------------- runner
def default_workers():
    n = os.cpu_count() or 2
    return max(1, min(n - 1, 8))


class PoolRunner(QtCore.QObject):
    """Qt front end of the pool. submit() takes a list of (function, args) tasks; each task
    returns a list of (key, result, seconds) which arrive as `partial(gen, key, (result, s))`;
    `done(gen, None)` fires when all tasks of the generation are back."""
    partial = QtCore.pyqtSignal(int, str, object)
    done = QtCore.pyqtSignal(int, object)
    failed = QtCore.pyqtSignal(int, str)
    progressed = QtCore.pyqtSignal(int, int, int)
    logged = QtCore.pyqtSignal(int, str)
    busy = QtCore.pyqtSignal(bool)
    _ret = QtCore.pyqtSignal(int, object)
    _err = QtCore.pyqtSignal(int, str)

    _shared = None       # one pool for the whole application

    def __init__(self, parent=None, workers=None):
        super().__init__(parent)
        self.gen = 0
        self.outstanding = {}
        self.threads = set()             # API parity with Runner (always empty)
        self._ret.connect(self._on_ret)
        self._err.connect(self._on_err)
        self.workers = workers or default_workers()
        P = self.pool(self.workers)
        self.ch = P.register(self)

    # -- pool management (shared)
    @classmethod
    def pool(cls, workers=None):
        if cls._shared is None:
            cls._shared = _Pool(workers or default_workers())
        return cls._shared

    @classmethod
    def shutdown(cls):
        if cls._shared is not None:
            cls._shared.close()
            cls._shared = None

    @property
    def mode(self):
        return self.pool(self.workers).mode

    def submit(self, tasks, combine=None):
        """tasks: list of (fn, args-without-gen). combine: optional fn(list of task returns)
        -> object emitted with done() (used by Monte Carlo)."""
        P = self.pool(self.workers)
        gen = P.next_gen(self.ch)
        self.gen = gen
        self.outstanding = {"gen": gen, "n": len(tasks), "got": [], "combine": combine,
                            "t0": time.perf_counter()}
        if not tasks:
            self.done.emit(gen, None)
            return gen
        for fn, args in tasks:
            P.apply(fn, ((self.ch, gen),) + tuple(args),
                    lambda res, gen=gen: self._ret.emit(gen, res),
                    lambda e, gen=gen: self._err.emit(gen, f"{type(e).__name__}: {e}"))
        self.busy.emit(True)
        return gen

    def is_current(self, gen):
        return gen == self.gen

    def running(self):
        o = self.outstanding
        return bool(o) and len(o["got"]) < o["n"]

    def cancel(self):
        if self.running():
            P = self.pool(self.workers)
            P.next_gen(self.ch)                           # queued tasks become no-ops
            long = time.perf_counter() - self.outstanding["t0"] > 0.3
            self.drop()
            if long:
                P.restart()                               # kill long-running ones

    def drop(self, msg=None):
        if self.outstanding:
            gen = self.outstanding["gen"]
            self.outstanding = {}
            self.busy.emit(False)
            if msg:
                self.failed.emit(gen, msg)

    def _on_ret(self, gen, res):
        o = self.outstanding
        if not o or gen != o["gen"]:
            return
        o["got"].append(res)
        if res is not None and o["combine"] is None:
            for key, r, dt in res:
                self.partial.emit(gen, key, (r, dt))
        self.progressed.emit(gen, len(o["got"]), o["n"])
        if len(o["got"]) == o["n"]:
            out = None
            if o["combine"] is not None:
                try:
                    out = o["combine"]([g for g in o["got"] if g is not None])
                except Exception as e:
                    self.failed.emit(gen, f"{e}\n{traceback.format_exc()}")
            self.outstanding = {}
            self.busy.emit(False)
            self.done.emit(gen, out if o["combine"] is not None else time.perf_counter() - o["t0"])

    def _on_err(self, gen, msg):
        o = self.outstanding
        if not o or gen != o["gen"]:
            return
        o["got"].append(None)
        self.failed.emit(gen, msg)
        if len(o["got"]) == o["n"]:
            self.outstanding = {}
            self.busy.emit(False)
            self.done.emit(gen, None)

    def wait_idle(self, ms=600_000):
        from PyQt5 import QtWidgets
        app = QtWidgets.QApplication.instance()
        t = time.time()
        while self.running() and time.time() - t < ms / 1000:
            app.processEvents(); time.sleep(0.003)
        app.processEvents()


class _Pool:
    CHANNELS = 16

    def __init__(self, workers):
        self.workers = workers
        self.mode = "process"
        self._gens = [0] * self.CHANNELS
        self.runners = []
        self._start()

    def register(self, runner):
        self.runners.append(runner)
        return len(self.runners) - 1

    def _start(self):
        try:
            if os.environ.get("ESTTOOLBOX_THREADS"):
                raise RuntimeError("thread mode requested")
            ctx = mp.get_context("spawn")
            self.gen = ctx.Array("i", self._gens)
            self.p = ctx.Pool(self.workers, initializer=_init, initargs=(self.gen,))
            self.p.apply_async(_warm)
        except Exception:
            self.mode = "thread"
            global _GEN
            self.gen = _LocalArray(self._gens)
            _GEN = self.gen
            self.p = ThreadPool(max(2, self.workers))

    def next_gen(self, ch):
        self._gens[ch] += 1
        with self.gen.get_lock():
            self.gen[ch] = self._gens[ch]
        return self._gens[ch]

    def apply(self, fn, args, cb, ecb):
        self.p.apply_async(fn, args, callback=cb, error_callback=ecb)

    def restart(self):
        for r in self.runners:          # their tasks die with the pool
            r.drop("cancelled")
        old = self.p
        try:
            old.terminate()
        except Exception:
            pass
        self._start()

    def close(self):
        try:
            self.p.terminate()
            self.p.join()
        except Exception:
            pass


class _LocalArray(list):
    def __init__(self, v):
        import threading
        super().__init__(v)
        self._lock = threading.Lock()

    def get_lock(self):
        return self._lock


atexit.register(PoolRunner.shutdown)
