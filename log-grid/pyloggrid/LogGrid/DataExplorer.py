"""Data processing and visualisation"""

import logging
import os
import typing
from typing import Any, Callable, NotRequired, Optional, TypedDict, TypeVar

if typing.TYPE_CHECKING:  # for Sphinx
    import numpy as np

    np.zeros(0)

from pyloggrid.Libs.singlethread_numpy import np

np.zeros(0)  # Do not remove - used to force singlethread_numpy to be imported before the other libraries when refactoring

import joblib

from pyloggrid.Libs.custom_logger import setup_custom_logger
from pyloggrid.Libs.IOLib import add_drawables, load_step, read_json_settings
from pyloggrid.Libs.misc import TimeTracker
from pyloggrid.LogGrid.Grid import Grid

logger = setup_custom_logger(__name__, level=logging.INFO)


class DataExplorer:
    """
    Data processing and visualisation

    Args:
        data_path: simulation's folder
    """

    def __init__(self, data_path: str):
        self.path = data_path
        assert os.path.isdir(data_path), f"The path {data_path} could not be found"

        self.time_tracker = TimeTracker()
        self.time_tracker.start_timer("total")

        with self.time_tracker("disk"):
            settings = read_json_settings(data_path)

        self.D = settings["D"]
        self.l_params = settings["l_params"]
        self.simu_params = settings["simu_params"]
        self.fields_name = settings["fields_name"]
        self.N_steps = settings["N_steps"]

        logger.info(f"Expecting {self.N_steps} steps to load")

    def getgrid(self, N_pts: int, k_min: float, fields: dict[str, np.ndarray] = None, k0: bool = False) -> Grid:
        """Creates a grid populated with a step's fields.

        Args:
            N_pts: grid size
            k_min: grid's min k
            fields: grid's fields
            k0: whether grid has k=0

        Returns:
            the populated grid
        """
        grid = Grid(D=self.D, l_params=self.l_params, N_points=N_pts, fields_name=self.fields_name, k_min=k_min, k0=k0)
        if fields is not None:
            grid.load_fields(fields)
        return grid

    # noinspection PyMissingOrEmptyDocstring
    class DrawFuncDict(TypedDict):
        get: Callable[["DataExplorer"], dict] | Callable[[Grid, float, dict], dict]
        plot: Callable[[Callable], None]
        perframe: NotRequired[bool]

    def display(
        self,
        draw_funcs: dict[str, DrawFuncDict],
        N_points: Optional[int] = None,
        loadfromsave: bool = False,
        N_min: int = 1,
        N_max: Optional[int] = None,
        n_jobs: int = 1,
    ) -> None:
        """Compute and display all the requested drawables.

        Args:
            draw_funcs: functions to draw, as a dict ``{name -> {"get"->fun, "plot"->fun(drawables, ts), "perframe"->bool}}``.
                If ``perframe``, the getter is per step and has signature ``fun(grid, t, simu_params)``. Otherwise, it is called once with signature ``fun(dataExplorer)``
                If ``perframe`` is omitted, default to ``True``
            N_points: max number of points to evaluate. If ``None``, equal to the max available points
            N_min: The first step evaluated
            N_max: The last step evaluated
            loadfromsave: data is not calculated but directly loaded from already computed drawables in ``drawables.npy``. ``N_min, N_max, N_points`` are not taken into account.
            n_jobs: the number of parrallel threads to use. Use ``-1`` for unlimited (not recommended)
        """
        drawables = {}  # name -> data to draw
        if loadfromsave:  # check if previous data exists, and load if possible
            with self.time_tracker("disk"):
                if os.path.isfile(f"{self.path}/drawables.npy"):
                    logger.info(f"Loading data from {self.path}/drawables.npy")
                    # noinspection PyTypeChecker
                    dict_draw: dict = np.load(f"{self.path}/drawables.npy", allow_pickle=True).item()
                    drawables = dict_draw["drawables"]
                    nameNotInDrawables = []
                    for name, draw_func in draw_funcs.items():
                        if name not in drawables:
                            nameNotInDrawables.append(name)

                    if nameNotInDrawables:
                        logger.warning(f"No data found for {nameNotInDrawables}, recomputing")
                        loadfromsave = False
                else:
                    logger.warning(f"{self.path}/drawables.npy doesn't exist, setting loadfromsave=False")
                    loadfromsave = False

        if not loadfromsave:
            draw_funcs_filtered = {k: v for k, v in draw_funcs.items() if k not in drawables}

            logger.info("Warning: this assumes a divergence-free flow field")

            if N_max is None or N_max > self.N_steps:
                N_max = self.N_steps
            N_points = N_max - N_min + 1 if N_points is None else min(N_points, N_max - N_min + 1)

            draw_perframe = []
            draw_notperframe = []
            for name, draw_func in draw_funcs_filtered.items():
                if ("perframe" not in draw_func) or draw_func["perframe"]:
                    draw_perframe.append([name, draw_func])
                    drawables[name] = []
                else:
                    draw_notperframe.append([name, draw_func])

            def load_time_dependant_drawables(step: int) -> tuple[dict[str, Any], TimeTracker]:
                """
                Load all the time-dependant variables for the given step
                :param step: step number
                :return: dict drawable_name -> drawable result
                """
                logger.info(f"Loading step {step}/{self.N_steps} ({np.round(100 * step / self.N_steps, 2)}%)")
                drawables_perstep = {}
                self.time_tracker = TimeTracker()  # reset and return because parallelization loses the data
                self.time_tracker.start_timer("total")

                # load step data
                with self.time_tracker("disk"):
                    item: dict = self.load_step(step, grid=False)
                t, N_pts, k_min, fields, k0 = item["t"], item["N_points"], item["k_min"], item["fields"], item["k0"]

                # create corresponding grid
                grid = self.getgrid(N_pts, k_min, fields, k0)
                # compute each drawable
                with self.time_tracker("drawables_perstep"):
                    for name, draw_func in draw_perframe:
                        drawables_perstep[name] = draw_func["get"](grid, t, self.simu_params)
                        drawables_perstep[name]["t"] = t

                self.time_tracker.end_timer("total")
                return drawables_perstep, self.time_tracker

            # compute all the time-dependant quantities
            if draw_perframe:
                self.time_tracker.end_timer("total")  # stop counting during parallel step to avoid double counting
                res_parallel = joblib.Parallel(n_jobs=n_jobs)(
                    joblib.delayed(load_time_dependant_drawables)(step) for step in np.round(np.linspace(N_min, N_max, N_points)).astype(int)
                )
                self.time_tracker.start_timer("total")
                for step, (drawable_step, timer_step) in enumerate(res_parallel):
                    # append each data part to their data holder
                    for k, v in timer_step.elapsed_time.items():
                        if k not in self.time_tracker.elapsed_time:
                            self.time_tracker.elapsed_time[k] = 0
                        self.time_tracker.elapsed_time[k] += v
                    for name, draw_func in draw_perframe:
                        drawables[name].append(drawable_step[name])

                # re-aggregate the drawables by label:
                # original: [{"a": 0, "b": 5}, {"a": 2, "b": -1}]
                # new: {"a": [0, 2], "b": [5, -1]}
                for name, _ in draw_perframe:
                    drawables_bk = np.array(drawables[name])
                    drawables[name] = {}
                    for key in drawables_bk[0].keys():
                        drawables[name][key] = []
                    for i in range(N_points):
                        for k, v in drawables_bk[i].items():
                            drawables[name][k].append(v)
                    for key in drawables_bk[0].keys():
                        try:
                            drawables[name][key] = np.array(drawables[name][key])
                        except ValueError:
                            drawables[name][key] = np.array(drawables[name][key], dtype="object")

            # compute global quantities
            with self.time_tracker("drawables_global"):
                for name, draw_func in draw_notperframe:
                    logger.info(f"Calculating {name}")
                    drawables[name] = draw_func["get"](self)

            # Save all data as npy
            with self.time_tracker("disk"):
                add_drawables(self.path, drawables)

        # Plot
        self.time_tracker.end_timer("total")
        logger.info(f"Done ! Now plotting - elapsed: {self.time_tracker}")
        for name, draw_func in draw_funcs.items():
            # noinspection PyUnboundLocalVariable
            draw_func["plot"](lambda *args: ((drawables[name][i] for i in args) if len(args) > 1 else drawables[name][args[0]]))

    def load_step(self, step: int = None, ts: np.ndarray = None, t: float = None, grid: bool = True) -> dict | tuple[dict, Grid]:
        """
        Load a saved step, either by step number, or by simulation time. If time is provided, it overrides the step.

        Args:
            step: step to load
            ts: array of simulation times
            t: time of the step to use. Must be provided with ``ts``. Overrides ``step``.
            grid: if ``True``, create a grid from loaded data and return it

        Returns:
            step data ``{fields, t, N_points, k_min, elapsed_time}`` if not ``grid``, else a tuple ``step_data, grid``
        """
        if t is not None:
            assert ts is not None, "`ts` must be provided with t"
            step_resc = np.argmax(ts >= t)
            step = int(step_resc * self.N_steps / ts.size)
            logger.info(f"Loading timed step: {step}, t={t}")
        elif step is None:
            step = self.N_steps

        data = load_step(fields_names=self.fields_name, path=self.path, step=step)
        if not grid:
            return data
        N_points, fields, k_min = data["N_points"], data["fields"], data["k_min"]
        return data, self.getgrid(N_points, k_min, fields)


T = TypeVar("T", bound=str)
PlotFun = Callable[..., tuple[np.ndarray, ...]]
