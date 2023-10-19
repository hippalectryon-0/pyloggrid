"""Solver for the log grids."""
from __future__ import annotations

import typing

if typing.TYPE_CHECKING:  # for Sphinx
    import numpy as np

    np.zeros(0)

from pyloggrid.Libs.singlethread_numpy import np

np.zeros(0)  # Do not remove - used to force singlethread_numpy to be imported before the other libraries when refactoring

import contextlib
import datetime
import logging
import os
import shutil
import sys
import time
from typing import Callable, Optional, Union

from rkstiff import etd35

from pyloggrid.Libs.custom_logger import setup_custom_logger
from pyloggrid.Libs.IOLib import load_step, read_json_settings, save_step, update_json_settings
from pyloggrid.Libs.misc import TimeTracker
from pyloggrid.Libs.singlethread_numpy import np
from pyloggrid.LogGrid.Grid import Grid

logger = setup_custom_logger(__name__, level=logging.INFO)


## A few useful functions


def fields_to_1D(fields: dict[str, np.ndarray], field_names: list[str]) -> np.ndarray:
    """
    Convert N-dimensional fields to 1D arrays

    Args:
        field_names: [name1, ...] name of the fields (order is important, as it it used to convert back from 1D)
        fields: N-dimensional named fields, grid-shaped

    Returns:
        1D ndarray containing all the appended 1D fields, as ordered by ``field_names``
    """
    each_field_linear = np.array([fields[arr_name].reshape(fields[arr_name].size) for arr_name in field_names])
    return each_field_linear.reshape(each_field_linear.size)


def D1_to_fields(linear: np.ndarray, field_names: list[str], field_shape: (int, ...)) -> dict[str, np.ndarray]:
    """Reverse of :func:`fields_to_1D`

    Converts one 1D array to a dict of the encoded fields in their original shape

    Args:
        linear: 1D ndarray of fields
        field_names: [name1, ...] name of the fields (order is important)
        field_shape: shape of the grid / fields

    Returns:
        the fields with their full shape
    """
    linears_with_fields = linear.reshape((len(field_names),) + field_shape)
    return {field_names[i]: linears_with_fields[i].reshape(field_shape) for i in range(len(field_names))}


class CustIntegrator:
    """Abstract integrator class"""

    def __init__(
        self,
        equation_nl: Callable[[float, np.ndarray], np.ndarray],
        equation_l: Callable[[float, np.ndarray], np.ndarray],
        save_step: Callable[[float, float, np.ndarray], bool],
        init_t: float,
        solver_params: dict,
        y0: np.ndarray,
    ):
        self.equation_nl = equation_nl
        self.equation_l = equation_l
        self.save_step = save_step
        self.init_t = init_t

        self.y = np.array(y0)
        self.t = self.init_t
        self.status = "not started"
        self.stopsim = False  # to stop the simulation

        self.minh = solver_params.get("minh", 1e-16)

    def select_initial_step(self, t0: float, y0: np.ndarray, order: int, rtol: float = 1e-6, atol: float = 1e-6) -> float:
        """Empirically select a good initial step.
        The algorithm is described in [#]_.

        Args:
            t0: Initial value of the independent variable.
            y0: Initial value of the dependent variable.
            order: Error estimator order. It means that the error controlled by the algorithm is proportional to ``step_size ** (order + 1)``.
            rtol: Desired relative tolerance.
            atol: Desired absolute tolerance.

        Returns:
            ``h_abs`` = Absolute value of the suggested initial step.

        References:
            .. [#] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential Equations I: Nonstiff Problems", Sec. II.4.
        """
        norm = lambda x: np.linalg.norm(x) / x.size**0.5
        direction = 1
        if y0.size == 0:
            return np.inf

        scale = atol + np.abs(y0) * rtol
        scale_valid = np.abs(scale) != 0
        h0 = 1e-6
        f0 = self.equation_nl(t0, y0)
        dd0 = np.zeros(scale.shape, dtype="complex")
        dd0[scale_valid] = y0[scale_valid] / scale[scale_valid]
        d0 = norm(dd0)
        dd1 = np.zeros(dd0.shape, dtype="complex")
        dd1[scale_valid] = f0[scale_valid] / scale[scale_valid]
        d1 = norm(dd1)
        if d0 >= 1e-5 and d1 >= 1e-5:
            h0 = max(0.01 * d0 / d1 if np.isfinite(d1) else self.minh, self.minh)
        y1 = y0 + h0 * direction * f0
        f1 = self.equation_nl(t0 + h0 * direction, y1)
        dd2 = np.zeros(scale.shape, dtype="complex")
        dd2[scale_valid] = (f1 - f0)[scale_valid] / scale[scale_valid]
        d2 = norm(dd2) / h0

        if d1 <= 1e-15 and d2 <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            h1 = max((0.01 / max(d1, d2)) ** (1 / (order + 1)) if np.isfinite(d2) else self.minh, self.minh)

        return min(100 * h0, h1)

    # noinspection PyMissingOrEmptyDocstring
    def solve(self):
        raise NotImplementedError


class ViscDopri(CustIntegrator):
    """DOPRI5-based solver made specifically for log grids.

    A nonlinear update step is decoupled from the linear update step.
    Special care is given to handling numerical (float-induced) random numerical errors which yield huge gradients.
    Based on scipy's DOPRI5 implementation

    Args:
        equation_nl: linear-less update step
        equation_l: linear update step
        init_t: initial time
        y0: initial 1D array
        solver_params: = ``{"atol"[absolute tolerance]: 1e-6, "rtol"[relative tolerance]: 1e-4, "exact_viscous_splitting"[if True, use exact viscous splitting]: False}``
        dt_params: ``{dt0 = initial time step, dtmin, dtmax}``
    """

    def __init__(
        self,
        equation_nl: Callable[[float, np.ndarray], np.ndarray],
        equation_l: Callable[[float, np.ndarray], np.ndarray],
        save_step: Callable[[float, float, np.ndarray], bool],
        init_t: float,
        solver_params: dict,
        y0: np.ndarray,
        dt_params: dict[str, Optional[float]],
    ):
        super().__init__(equation_nl, equation_l, save_step, init_t, solver_params, y0)
        self.atol = solver_params.get("atol", 1e-6)
        self.rtol = solver_params.get("atol", 1e-4)
        self.exact_viscous_splitting = solver_params.get("exact_viscous_splitting", False)

        # Internal RK & DOPRI parameters
        self.K = np.empty((7, self.y.size), dtype=self.y.dtype)
        self.max_factor = 10
        self.min_factor = 0.2
        self.min_step = 1e-200
        self.error_exponent = -1 / 5
        self.safety = 0.9
        self.A = np.array(
            [
                [0, 0, 0, 0, 0],
                [1 / 5, 0, 0, 0, 0],
                [3 / 40, 9 / 40, 0, 0, 0],
                [44 / 45, -56 / 15, 32 / 9, 0, 0],
                [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0],
                [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
            ]
        )
        self.B = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])
        self.C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1])
        self.E = np.array([-71 / 57600, 0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525, 1 / 40])

        # Initial step
        self.dt = dt_params["dt0"] if "dt0" in dt_params else self.select_initial_step(self.t, self.y, 4, self.rtol, self.atol)
        self.dtmin, self.dtmax = dt_params.get("dtmin"), dt_params.get("dtmax")
        logger.info(f"Initial step: dt={self.dt}")

    def rk_step(
        self, fun: Callable[[float, np.ndarray], np.ndarray], t: float, y: np.ndarray, h: float, A: np.ndarray, B: np.ndarray, C: np.ndarray, K: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform a single Runge-Kutta step.
        This function computes a prediction of an explicit Runge-Kutta method and
        also estimates the error of a less accurate method.
        Notation for Butcher tableau is as in [#]_.

        Args:
            fun: Right-hand side of the system.
            t: Current time.
            y: Current state.
            h: Step to use.
            A: Coefficients for combining previous RK stages to compute the next stage. For explicit methods the coefficients at and above the main diagonal are zeros.
            B: Coefficients for combining RK stages for computing the final prediction.
            C: Coefficients for incrementing time for consecutive RK stages. The value for the first stage is always zero.
            K: Storage array for putting RK stages here. Stages are stored in rows. The last row is a linear combination of the previous rows with coefficients

        Returns: tuple ``(y_new, f_new)`` where ``y_new`` is the solution at ``t + h`` computed with a higher accuracy, and ``f_new`` is the derivative ``fun(t + h, y_new)``.

        References:
            .. [#] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential Equations I: Nonstiff Problems", Sec. II.4.
        """

        if self.exact_viscous_splitting:
            fun_old = fun

            # noinspection PyMissingOrEmptyDocstring
            def fun_new(t_, y_):
                dt = t_ - t
                visc = self.equation_l(t_, y_)
                return fun_old(t_, y_) * np.exp(-visc * dt)

            fun = fun_new

        K[0] = fun(t, y)
        for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
            dy = np.dot(K[:s].T, a[:s]) * h
            K[s] = fun(t + c * h, y + dy)

        y_new = y + h * np.dot(K[:-1].T, B)
        f_new = fun(t + h, y_new)

        K[-1] = f_new

        return y_new, f_new

    def rkdp(self, force_step: bool = False) -> tuple[np.ndarray, float, float]:
        """Performs one RKDP step"""

        # quick access
        atol, rtol = self.atol, self.rtol
        t = self.t
        y = self.y
        A, B, C, E = self.A, self.B, self.C, self.E

        h = max(self.dt, self.min_step)

        # try to step once
        step_accepted, step_rejected = False, False
        y_new, derivative_new, newh = None, None, None
        while not step_accepted:
            assert h >= self.min_step, f"Could not converge (step too small {h})"

            # force step if dt is too small compared to float precision. Not a great practice, but can "free" the stepper from tough landscapes.
            forced_step = False
            if self.dtmax is not None and h > self.dtmax:
                h = self.dtmax
            elif self.dtmin is not None and h < self.dtmin:
                h = self.dtmin
                forced_step = True
            if force_step:
                if h < t * 1e-15:
                    h = max(h, 3e-16 * t)
                    logger.warning(f"Forcing step, h/t={h / t}")
                    forced_step = True
            else:
                assert t + h > t, f"Can not add step {h} to time t={t}"

            # don't store derivative from prev step, because the linear step changes it in an unknown way
            y_new, derivative_new = self.rk_step(self.equation_nl, t, y, h, A, B, C, self.K)

            # Compute error
            scale = atol * max(np.max(np.abs(y)), np.max(np.abs(y_new))) + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            # remove too small values
            absurd_vals = scale == 0
            scale[absurd_vals] = 1
            RK_diff = np.abs(np.dot(self.K.T, E))
            RK_diff[absurd_vals] = 0
            error_norm = np.max(RK_diff * h / np.abs(scale))  # was np.norm
            # Check error value, and adjust time steps
            if error_norm < 1 or forced_step:
                step_accepted = True
                if error_norm == 0:
                    factor = self.max_factor
                else:
                    # noinspection PyTypeChecker
                    factor = max(self.min_factor, min(self.max_factor, self.safety * error_norm**self.error_exponent))

                if step_rejected:
                    factor = min(1.0, factor)

                newh = h * factor
            else:
                step_rejected = True
                h *= max(self.min_factor, self.safety * error_norm**self.error_exponent)

            if forced_step:
                newh = max(t / 1e15 / 3, newh)
        # Perform linear update
        visc = self.equation_l(t, y_new)
        y_new = y * np.exp(visc * h) + (y_new - y) * np.sum([np.exp(visc * h * (1 - c)) * b for b, c in zip(self.B, self.C)], axis=0)
        # y_new = y_new * np.exp(visc * h)  # old way of updating: more stable but less accurate
        self.stopsim = self.save_step(t + h, h, y_new)

        return y_new, h, newh

    def solve(self) -> None:
        """Main solving loop.

        The solver runs until an interruption is fired by the update."""
        self.status = "running"
        while not self.stopsim:
            newy, useddt, newdt = self.rkdp()
            self.y = newy
            self.dt = newdt
            self.t += useddt

        self.status = "finished"


class ETD4RK(CustIntegrator):
    r"""Based on `Exponential Time Differencing for Stiff Systems | Elsevier Enhanced Reader`, Cox & Matthews, 2001, equations (26)-(29).

    Warning:
        Only designed for constant-in-time viscosities

    Args:
        equation_nl: linear-less update step
        equation_l: linear update step
        init_t: initial time
        y0: initial 1D array
        dt_params: ``{dt0 = initial time step, dtmin, dtmax}``
    """

    def __init__(
        self,
        equation_nl: Callable[[float, np.ndarray], np.ndarray],
        equation_l: Callable[[float, np.ndarray], np.ndarray],
        save_step: Callable[[float, float, np.ndarray], bool],
        init_t: float,
        y0: np.ndarray,
        dt_params: dict[str, Optional[float]],
    ):
        super().__init__(equation_nl, equation_l, save_step, init_t, {}, y0)

        # Initial step
        self.dt = dt_params["dt0"]
        logger.info(f"Initial step: dt={self.dt}")

    @staticmethod
    def dopri_step(
        fun: Callable[[float, np.ndarray], np.ndarray], t: float, y: np.ndarray, h: float, A: np.ndarray, B: np.ndarray, C: np.ndarray, K: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform a single Runge-Kutta step.
        This function computes a prediction of an explicit Runge-Kutta method and
        also estimates the error of a less accurate method.
        Notation for Butcher tableau is as in [#]_.

        Args:
            fun: Right-hand side of the system.
            t: Current time.
            y: Current state.
            h: Step to use.
            A: Coefficients for combining previous RK stages to compute the next stage. For explicit methods the coefficients at and above the main diagonal are zeros.
            B: Coefficients for combining RK stages for computing the final prediction.
            C: Coefficients for incrementing time for consecutive RK stages. The value for the first stage is always zero.
            K: Storage array for putting RK stages here. Stages are stored in rows. The last row is a linear combination of the previous rows with coefficients

        Returns: tuple ``(y_new, f_new)`` where ``y_new`` is the solution at ``t + h`` computed with a higher accuracy, and ``f_new`` is the derivative ``fun(t + h, y_new)``.
        References:
            .. [#] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential Equations I: Nonstiff Problems", Sec. II.4.
        """
        K[0] = fun(t, y)
        for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
            dy = np.dot(K[:s].T, a[:s]) * h
            K[s] = fun(t + c * h, y + dy)

        y_new = y + h * np.dot(K[:-1].T, B)
        f_new = fun(t + h, y_new)

        K[-1] = f_new

        return y_new, f_new

    def rk(self) -> np.ndarray:
        """Performs one Rk step"""
        # quick access
        t = self.t
        y = self.y  # u_n
        h = self.dt
        visc = self.equation_l(t, y)  # c
        hc = h * visc
        e = np.exp(hc / 2)
        e2 = np.exp(hc)
        F = self.equation_nl

        # perform RK. We need to take care of hc<<1 terms, where the precision of the exp becomes too bad to cancel Taylor terms
        F0 = F(t, y)
        a = np.zeros_like(y)
        mask0 = np.abs(hc) > 1e-6
        a[mask0] = y[mask0] * e[mask0] + (e[mask0] - 1) * F0[mask0] / visc[mask0]  # (26)
        a[~mask0] = y[~mask0] * e[~mask0] + F0[~mask0] * h / 2
        Fa = F(t + h / 2, a)
        b = np.zeros_like(y)
        b[mask0] = y[mask0] * e[mask0] + (e[mask0] - 1) * Fa[mask0] / visc[mask0]  # (27)
        b[~mask0] = y[~mask0] * e[~mask0] + Fa[~mask0] * h / 2
        Fb = F(t + h / 2, b)
        c = np.zeros_like(y)
        c[mask0] = a[mask0] * e[mask0] + (e[mask0] - 1) * (2 * Fb[mask0] - F0[mask0]) / visc[mask0]  # (28)
        c[~mask0] = a[~mask0] * e[~mask0] + (2 * Fb[~mask0] - F0[~mask0]) * h / 2
        Fc = F(t + h, c)

        el0, el1, el2, y_new = np.zeros_like(y), np.zeros_like(y), np.zeros_like(y), np.zeros_like(y)
        mask1 = np.abs(hc) > 1e-3
        el0[mask1] = F0[mask1] * (-4 - hc[mask1] + e2[mask1] * (4 - 3 * hc[mask1] + hc[mask1] ** 2)) / (h**2 * visc[mask1] ** 3)
        el0[~mask1] = F0[~mask1] * h / 6
        el1[mask1] = 2 * (Fa[mask1] + Fb[mask1]) * (2 + hc[mask1] + e2[mask1] * (-2 + hc[mask1])) / (h**2 * visc[mask1] ** 3)
        el1[~mask1] = 2 * (Fa[~mask1] + Fb[~mask1]) * h / 6
        mask2 = np.abs(hc) > 6e-3
        el2[mask2] = Fc[mask2] * (-4 - 3 * hc[mask2] - hc[mask2] ** 2 + e2[mask2] * (4 - hc[mask2])) / (h**2 * visc[mask2] ** 3)
        el2[~mask2] = Fc[~mask2] * h / 6
        y_new = y * e2 + el0 + el1 + el2  # (29)

        self.stopsim = self.save_step(t + h, h, y_new)

        return y_new

    def solve(self) -> None:
        """Main solving loop. The solver runs until an interruption is fired by the update."""
        self.status = "running"
        while not self.stopsim:
            newy = self.rk()
            self.y = newy
            self.t += self.dt

        self.status = "finished"


class ETD35(CustIntegrator):
    """Based on https://github.com/whalenpt/rkstiff

    Warning:
        Only supports diagonal (and constant-in-time) linear terms for now

    Args:
        equation_nl: linear-less update step
        equation_l: linear update step
        init_t: initial time
        solver_params: ``{"rtol"[relative tolerance]: 1e-2, "adapt_cutoff"[Limits values used in the computation of the suggested step size to those with |u| > adapt_cutoff*max(|u|)]: 1e-2, "minh"[minimum time step]: 1e-16}``
        y0: initial 1D array
        dt_params: ``{dt0 = initial time step, dtmin, dtmax}``
    """

    def __init__(
        self,
        equation_nl: Callable[[float, np.ndarray], np.ndarray],
        equation_l: Callable[[float, np.ndarray], np.ndarray],
        save_step: Callable[[float, float, np.ndarray], bool],
        init_t: float,
        solver_params: dict,
        y0: np.ndarray,
        dt_params: dict[str, Optional[float]],
    ):
        super().__init__(equation_nl, equation_l, save_step, init_t, solver_params, y0)
        self.rtol = solver_params.get("rtol", 1e-2)
        self.adapt_cutoff = solver_params.get("adapt_cutoff", 1e-2)

        # Initial step
        self.dt = dt_params["dt0"] if "dt0" in dt_params else self.select_initial_step(self.t, self.y, 4)
        self.dt_suggested = self.dt
        logger.info(f"Initial step: dt={self.dt}")

    def rk(self) -> tuple[np.ndarray, float, float]:
        """Performs one Rk step

        Returns:
            tuple ``(y_new, h, h_new_suggested)``
        """
        # quick access
        t = self.t
        y = self.y

        solver = etd35.ETD35(
            linop=self.equation_l(t, y), NLfunc=lambda y_: self.equation_nl(t, y_), epsilon=self.rtol, adapt_cutoff=self.adapt_cutoff, minh=self.minh
        )
        y_new, h, h_new_suggested = solver.step(y, self.dt_suggested)

        self.stopsim = self.save_step(t + h, h, y_new)
        return y_new, h, h_new_suggested

    def solve(self) -> None:
        """Main solving loop. The solver runs until an interruption is fired by the update."""
        self.status = "running"
        while not self.stopsim:
            self.y, self.dt, self.dt_suggested = self.rk()
            self.t += self.dt

        self.status = "finished"


class SolverInterruptedError(InterruptedError):
    """custom error thrown to change grid size"""

    ...


class Solver:
    """Generic top-level object to handle solving log-grid equations

    Args:
        fields_names: the names of the grid-shaped fields
        equation_nl: the function that performs the linear-less update step. Returns the time derivative of the fields
        D: space dimension
        l_params: parameters that define the grid parameter `̀`l`` as a dict with keys ``{"a", "b", "plastic"}``. "plastic" supercedes all other. For ``l=2``, chose ``a=b=None``
        k_min: minimum k of the grid. Default is defined by ``Grid``.
        simu_params: physical quantities relevant to the simulation, fixed for the whole simulation.
        n_threads: Number of threads to use, default is max/2 (not recommended, run benchmarking for better results. If running batch simulations, 1 is optimal.)
        equation_l: implicit visocsity update step, called after the RK step has ended. Returns the new fields and their new time derivative
        k0: whether there's a k0 mode"""

    def __init__(
        self,
        fields_names: list[str],
        equation_nl: Callable[[float, Grid, dict], dict],
        D: int,
        l_params: dict[str, Union[float, bool]],
        equation_l: Callable[[float, Grid, dict], dict[str, np.ndarray]],
        k_min: float = None,
        simu_params: dict = None,
        n_threads: int = None,
        k0: bool = False,
    ):
        self.equation_nl = equation_nl
        self.equation_l = equation_l
        self.simu_params = simu_params
        self.end_simulation = None

        # Initialize internal counters
        self.step = 0  # current saved step. Equal to ode_step if we save every step
        self.ode_step = 0  # current simulation step (+1 for every true [= not rejected] step). Not equal to number of calls to equation_nonlinear, but equal to number of calls to equation_linear.
        self.time_tracker = TimeTracker()
        self.time_tracker.start_timer("total")
        self.grid = Grid(D=D, l_params=l_params, N_points=1, k_min=k_min, fields_name=fields_names, n_threads=n_threads, k0=k0)
        self.grid.time_tracker = self.time_tracker

    def load_parameters(self, path: str) -> tuple[float, float]:
        """Loads simulation parameters from settings.json.

        Overwrites instance variables

        Args:
            path: directory to load the settings from

        Returns:
            physical time of the simulation's last step and timestep
        """
        settings = read_json_settings(path)

        # Overwrite instance variables
        D = settings["D"]
        self.simu_params = settings["simu_params"]
        self.end_simulation = settings["end_simulation"]
        l_params = settings["l_params"]
        fields_name = settings["fields_name"]

        # Load last data step & overwrite relevant instance variables
        step = settings["N_steps"]
        ode_step = settings["N_ode_steps"]
        step_data = load_step(path=path, step=step, fields_names=fields_name)
        k0 = step_data["k0"]
        self.step = step
        self.ode_step = ode_step
        self.time_tracker = TimeTracker({k[7:]: v for k, v in step_data.items() if k[:7] == "ttrack_"})
        self.time_tracker.start_timer("total")
        N_points = next(iter(step_data["fields"].values())).shape[0] - (1 if k0 else 0)
        self.grid = Grid(
            D=D, l_params=l_params, N_points=N_points, k_min=step_data["k_min"], fields_name=fields_name, n_threads=self.grid.maths.n_threads, k0=k0
        )
        self.grid.fields = step_data["fields"]
        self.grid.time_tracker = self.time_tracker

        return float(step_data["t"]), float(step_data["dt"])

    def save_step_all(self, t: float, dt: float, path: str) -> None:
        """Save the current step. Saves both fields and settings.

        Args:
            t: current time
            dt: timestep
            path: folder to save to. Will override any existing file.
        """
        self.step += 1
        logger.info(f"Saving real step {self.step}, t={t:8}")

        save_step(path, step=self.step, grid=self.grid, t=t, dt=dt, time_tracker=self.time_tracker)
        update_json_settings(path, {"N_steps": self.step, "N_ode_steps": self.ode_step}, update=True)

    def solve(
        self,
        save_path: str,
        initial_conditions: str | Callable[[dict[str, np.ndarray], Grid, dict], dict[str, np.ndarray]] | tuple | list,
        solver_params: dict = None,
        init_t: float = 0,
        end_simulation=None,
        save_one_in: float = 1,
        update_gridsize_cb: Optional[Callable[[Grid], Optional[int]]] = None,
        dt_params: dict[str, Optional[float]] = None,
        solver: typing.Literal["ViscDopri", "ETD4RK", "ETD35"] = "ETD35",
    ) -> None:
        """Solve the solver's equation with the required numerical parameters.

        Args:
            save_path: where to save the fields and simulation parameters. Will backup if exists, will create otherwise
            initial_conditions: if str: either ``"loadfromsave"`` to resume simulation, [legacy: or the path of the ``.npz`` save file whence to initialize the fields].
                If tuple ``(name, step)``: load step ``step`` from .h5 ``name``.
                Otherwise: function that returns the initial fields.
                If "loadfromsave", it will override all dependant settings: ``end_simulation, rtol, atol, l_params, D, fields_name, simu_params, grid, k_min``
            solver_params: params forwarded to the solver
            init_t: initial simulation time
            end_simulation: dict of thresholds to end the simulation: ``{t: max physical time, elapsed_time: max real time spent computing, step: max save step}``
            save_one_in: save one step to file every X ``ode_step``
            update_gridsize_cb: optional callback to change the grid size after a step. Returns ``None`` if no change is to be made, returns the new grid size otherwise.
            dt_params: ``{dt0: initial timestep, dtmin, dtmax}``
            solver: among ``"ETD35"`` (default), ``"ETD4RK"``, ``"ViscDopri"``
        """
        if solver_params is None:
            solver_params = {}
        if end_simulation is None:
            end_simulation = {}
        end_simulation = {
            "t": end_simulation["t"] if "t" in end_simulation else None,
            "elapsed_time": end_simulation["elapsed_time"] if "elapsed_time" in end_simulation else None,
            "step": end_simulation["step"] if "step" in end_simulation else None,
            "ode_step": end_simulation["ode_step"] if "ode_step" in end_simulation else None,
        }
        self.end_simulation = end_simulation

        # Initial conditions (IC)
        if callable(initial_conditions):
            fields = initial_conditions(self.grid.fields, self.grid, self.simu_params)
            self.grid = self.grid.to_new_size(fields)
        elif initial_conditions == "loadfromsave":
            params = {"end_simulation": end_simulation, "solver_params": solver_params, "simu_params": self.simu_params}  # override previous parameters
            update_json_settings(save_path, params, update=True)

            init_t, dt = self.load_parameters(save_path)
            if dt > 0:  # dt=0 if only the initial step is saved
                if dt_params is None:
                    dt_params = {}
                dt_params["dt0"] = dt
            logger.info(f"Starting again from t={init_t}, dt={dt:.2e}, step={self.step}, grid size N={self.grid.N_points}")
        else:
            if isinstance(initial_conditions, (tuple, list)):
                logger.info(f"Loading IC from: {initial_conditions[0]} step {initial_conditions[1]}")
                step_data = load_step(self.grid.fields_names, filepath=initial_conditions[0], step=initial_conditions[1])
            elif isinstance(initial_conditions, str):  # load from a given .npz step, legacy method
                logger.info(f"Loading IC from: {initial_conditions}")
                step_data = load_step(filepath=initial_conditions, fields_names=self.grid.fields_names)
            else:
                raise ValueError("Wrong initial_conditions provided")

            self.grid = Grid(
                D=self.grid.D,
                l_params=self.grid.l_params,
                N_points=step_data["N_points"],
                k_min=step_data["k_min"],
                fields_name=self.grid.fields_names,
                n_threads=self.grid.maths.n_threads,
                k0=step_data["k0"],
            )
            self.grid.fields = step_data["fields"]
            self.grid.time_tracker = self.time_tracker

        for field_name in self.grid.fields:  # make sure someone didn't forget to cast as complex in init conditions...
            self.grid.fields[field_name] = self.grid.fields[field_name].astype(complex)

        def equation_nl_convertdim(t: float, y: np.ndarray) -> np.ndarray:
            """ "Man-in-the-middle" update step that converts N-D <-> 1D.

            Args:
                t: time
                y: 1D DOPRI array

            Returns:
                the 1D result of the update step
            """

            # 1D -> N-D
            new_fields = D1_to_fields(y, self.grid.fields_names, self.grid.shape)
            self.grid.fields = new_fields  # sets grid fields to current values

            # N-D -> 1D
            return fields_to_1D(self.grid.maths.enforce_grid_symmetry_dict(self.equation_nl(t, self.grid, self.simu_params)), self.grid.fields_names)

        def save_step(t: float, dt: float, y: np.ndarray) -> bool:
            """Save the current step and check if the simulation should end
            Also enforces symmetry if needed

            Args:
                t: time
                dt: time step used
                y: 1D DOPRI array

            Returns:
                whether the sim should end
            """
            newy = D1_to_fields(y, self.grid.fields_names, self.grid.shape)

            self.grid.fields = newy  # sets grid fields to current values
            self.grid.enforce_grid_symmetry()

            # update solver parameters
            self.ode_step += 1
            self.time_tracker.tick_timer("total")
            logger.debug(f"Solver: ode step {self.ode_step}, t={t}, dt={dt}, elapsed time: {datetime.timedelta(seconds=self.time_tracker.get('total'))}")

            # Check whether to stop the simulation
            stopsim = False
            sst, sset, sss, ssos = self.end_simulation["t"], self.end_simulation["elapsed_time"], self.end_simulation["step"], self.end_simulation["ode_step"]
            if sst is not None and t >= sst:
                logger.info(f"Reached time: {sst}")
                stopsim = True
            elif sset is not None and self.time_tracker.get("total") >= sset:
                logger.info(f"Reached elapsed time: {sset}")
                stopsim = True
            elif sss is not None and self.step >= sss:
                logger.info(f"Reached saved step: {sss}")
                stopsim = True
            elif ssos is not None and self.ode_step >= ssos:
                logger.info(f"Reached ode step: {ssos}")
                stopsim = True

            # Check for grid size update
            with self.time_tracker("update_gridsize"):
                new_N_points = None
                if update_gridsize_cb is not None:
                    new_N_points = update_gridsize_cb(self.grid)
            if new_N_points is not None and not stopsim:
                self.time_tracker.start_timer("update_gridsize")
                logger.info(f"Updating the grid: old N = {self.grid.N_points}, new N = {new_N_points}")
                self.grid = self.grid.to_new_size_empty(new_N_points).load_fields(self.grid.fields)
                self.save_step_all(t, dt, save_path)

                logger.info("Starting new solver on updated grid")
                self.time_tracker.end_timer("update_gridsize")
                self.solve(
                    solver_params=solver_params,
                    initial_conditions="loadfromsave",
                    init_t=init_t,
                    save_path=save_path,
                    save_one_in=save_one_in,
                    end_simulation=end_simulation,
                    dt_params=dt_params,
                    update_gridsize_cb=update_gridsize_cb,
                    solver=solver,
                )
                raise SolverInterruptedError()

            # Save data
            with self.time_tracker("disk"):
                if stopsim or self.ode_step % save_one_in == 0:
                    self.save_step_all(t, dt, save_path)

            return stopsim

        def equation_l_convertdim(t: float, y: np.ndarray) -> np.ndarray:
            """Convert the linear array 1D <-> ND

            Args:
                t: time
                y: 1D DOPRI array

            Returns:
                1D new y, 1D new dy/dt
            """

            new_fields = D1_to_fields(y, self.grid.fields_names, self.grid.shape)
            self.grid.fields = new_fields
            visc = self.equation_l(t, self.grid, self.simu_params)

            return fields_to_1D(visc, self.grid.fields_names)

        # load / initialize parameters & save paths
        if initial_conditions != "loadfromsave":
            # Check for existing results
            if os.path.isdir(save_path):
                # Rename the folder just in case the user forgot to save it
                bk_name = f"{save_path}_bk_{int(time.time())}"
                os.rename(save_path, bk_name)
                logger.info(f"Save dir was not empty, renamed to {bk_name}")
                # make new folder
                os.makedirs(save_path)
            else:
                logger.info(f"Folder {save_path} does not exist, creating a new one")
                os.makedirs(save_path)

            # save current simulation file to keep a trace of what we ran
            orig_file = sys.modules["__main__"].__file__
            shutil.copyfile(orig_file, os.path.join(save_path, "source.py"))

            # initialize and save solver parameters
            params = {
                "init_t": init_t,
                "end_simulation": end_simulation,
                "N_steps": 0,
                "N_ode_steps": 0,
                "l_params": self.grid.l_params,
                "D": self.grid.D,
                "solver_params": solver_params,
                "fields_name": self.grid.fields_names,
                "simu_params": self.simu_params,
            }
            update_json_settings(save_path, params)
            init_t = params["init_t"]

        # Start solver
        logger.info("*** Starting solver ***")

        match solver:
            case "ViscDopri":
                ode = ViscDopri(
                    equation_nl=equation_nl_convertdim,
                    equation_l=equation_l_convertdim,
                    init_t=init_t,
                    solver_params=solver_params,
                    y0=fields_to_1D(self.grid.fields, self.grid.fields_names),
                    dt_params=dt_params if dt_params is not None else {},
                    save_step=save_step,
                )
            case "ETD4RK":
                ode = ETD4RK(
                    equation_nl=equation_nl_convertdim,
                    equation_l=equation_l_convertdim,
                    init_t=init_t,
                    y0=fields_to_1D(self.grid.fields, self.grid.fields_names),
                    dt_params=dt_params if dt_params is not None else {},
                    save_step=save_step,
                )
            case "ETD35":
                ode = ETD35(
                    equation_nl=equation_nl_convertdim,
                    equation_l=equation_l_convertdim,
                    init_t=init_t,
                    solver_params=solver_params,
                    y0=fields_to_1D(self.grid.fields, self.grid.fields_names),
                    dt_params=dt_params if dt_params is not None else {},
                    save_step=save_step,
                )
            case _:
                raise ValueError(f"Unknown solver provided: {solver}")
        if initial_conditions != "loadfromsave":
            self.save_step_all(init_t, 0, save_path)  # save first step
        with contextlib.suppress(SolverInterruptedError):  # Used to restart with a different grid
            ode.solve()
            logger.info(f"Finished sim. at t={ode.t}, success: {ode.status}, ode_steps: {self.ode_step}, saved steps:{self.step}, elapsed: {self.time_tracker}")
