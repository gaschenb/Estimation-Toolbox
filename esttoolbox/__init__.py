"""State Estimation Toolbox — state estimation, smoothing and parameter estimation.

    from esttoolbox import get_ssm, run_method, smooth, monte_carlo
    p = get_ssm("ct-radar"); model = p.model()
    xs, ys = model.simulate(p.T, rng=0)
    res = run_method("ukf", model, ys); sm = smooth(res, model)
"""
__version__ = "1.0.0"

from .core.systems import get_ssm, list_ssm, SSMProblem
from .core.ssm import StateSpaceModel, LinearGaussianSSM
from .core.run import STATE_METHODS, run_method, smooth, monte_carlo, applicable
from .core.static import get_static, list_static
from .core.paramest import METHODS as STATIC_METHODS
from .core import consistency, sysid
from .core.userdata import (load_series, load_static, series_from_arrays, kinematic_problem,
                            data_problems, full_truth)

__all__ = ["get_ssm", "list_ssm", "SSMProblem", "StateSpaceModel", "LinearGaussianSSM",
           "STATE_METHODS", "run_method", "smooth", "monte_carlo", "applicable",
           "get_static", "list_static", "STATIC_METHODS", "consistency", "sysid",
           "load_series", "load_static", "series_from_arrays", "kinematic_problem", "data_problems",
           "full_truth"]
