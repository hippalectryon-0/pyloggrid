""".. _IOLib:

A library to handle reading/writing files.
"""

from __future__ import annotations

from pyloggrid.Libs.singlethread_numpy import np

np.zeros(0)  # Do not remove - used to force singlethread_numpy to be imported before the other libraries when refactoring

import logging
import os
import pathlib
import shutil
import typing
from typing import Callable, List

import h5py as h5py
import orjson

from pyloggrid.Libs.custom_logger import setup_custom_logger
from pyloggrid.LogGrid.Grid import Grid

if typing.TYPE_CHECKING:
    from pyloggrid.Libs.misc import TimeTracker

logger = setup_custom_logger(__name__, level=logging.INFO)


# Decorators
def store_hdf5(fun: Callable) -> Callable:
    """Decorator to handle fail-safe saving to hdf5.

    Attempts to save to ``<path>/output.h5``, then to ``<path>/output_bk.h5``.

    Note:
        The decorated function's path argument becomes a directory instead of a file.

    Args:
        fun: a non failsafe saving function that takes a ``path`` (hdf5 file) first argument

    Example:
        ::

            @store_hdf5
            def add_group(path: str, group_name: str) -> None:
                path: h5py.File  # from the decorator
                path.create_group(group_name)
    """

    # noinspection PyMissingOrEmptyDocstring
    def wrapper(path: str | pathlib.Path, *args, **kwargs) -> None:
        path = pathlib.Path(path)
        normal_path = path / "output.h5"
        bk_path = path / "output_bk.h5"
        if normal_path.exists() and not bk_path.exists():
            shutil.copy(normal_path, bk_path)
        with h5py.File(normal_path, "a") as f:
            fun(f, *args, **kwargs)
        with h5py.File(bk_path, "a") as f:
            fun(f, *args, **kwargs)

    return wrapper


def load_hdf5(fun: Callable) -> Callable:
    """
    Decorator to handle fail-safe loading from hdf5.

    Attempts to load from ``<path>/output.h5``, or to ``<path>/output_bk.h5`` if the first one fails.

    Note:
        The decorated function's path argument becomes a directory instead of a file.

    Warning:
        If the decorated ``path`` argument is a file instead of a directory, this will only load this file (non fail-safe).

    Args:
        fun: a non failsafe loading function that takes a ``path`` (hdf5 file) first argument

    Example:
        ::

            @load_hdf5
            def load_group(f: h5py.File, group_name: str) -> dict:
                group = f[group_name]
                return {name: group[name][:] for name in group}
    """

    # noinspection PyMissingOrEmptyDocstring
    def wrapper(path, filepath=None, *args, **kwargs):
        if filepath is not None:
            with h5py.File(filepath, "r") as f:
                return fun(f, *args, **kwargs)

        path = pathlib.Path(path)
        try:
            with h5py.File(path / "output.h5", "r") as f:
                return fun(f, *args, **kwargs)
        except OSError:  # file is corrupted / missing
            with h5py.File(path / "output_bk.h5", "r") as f:
                return fun(f, *args, **kwargs)

    return wrapper


# Json
def read_json_settings(path: str) -> dict:
    """reads the ``settings.json`` simulation file (fail-safe), returns the data

    Args:
        path: directory where to find the file

    Returns:
        the json data as a dict

    Raises:
        AssertionError: if the path doesn't exist
    """

    sett_path = f"{path}/settings.json"
    sett_bk_path = f"{path}/settings.json_bk"
    assert os.path.isfile(sett_path) or os.path.isfile(sett_bk_path), "The path to load data from doesn't exist"

    logging.debug(f"Attempting to read JSON settings from {sett_path}")

    try:
        with open(sett_path, "r") as f:
            settings = orjson.loads(f.read())
    except Exception as e:
        logger.warning(f"Could not read settings file ({e}), falling back on {sett_bk_path}")
        with open(sett_bk_path, "r") as f:
            settings = orjson.loads(f.read())

    return settings


def update_json_settings(path: str, params: dict, update: bool = False) -> None:
    """Save simulation parameters to ``settings.json`` file.

    This is fail-safe though the use of ``settings_bk.json``.

    Args:
        path: directory to save the settings to
        params: parameters to save to file
        update: if True, then existing parameters will be updated according to existing keys in ``params``. Otherwise, settings are overwritten.
    """
    params = dict(params)  # copy
    sett_path = f"{path}/settings.json"
    sett_bk_path = f"{path}/settings.json_bk"
    sett_new_path = f"{path}/settings.json_new"

    if update:  # update existing parameters with keys in `params`
        with open(sett_path, "r") as f:
            data = orjson.loads(f.read())
        for k, v in params.items():
            data[k] = v
        params = data

    if os.path.isfile(sett_path):
        shutil.copy(sett_path, sett_bk_path)  # backup existing file in case the data may get corrupted
    with open(sett_new_path, "wb") as f:
        f.write(orjson.dumps(params, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY))
    # only rename once the data has been written
    os.replace(sett_new_path, sett_path)


# Hdf5
def load_step(fields_names: List[str], path: str = None, step: int = None, filepath: str = None) -> dict:
    """Load the data corresponding to a simulation step saved in ``output.h5``, or older legacy formats.

    Only the ``path`` and ``filepath`` arguments are necessary in the latest version.

    Args:
        filepath: path to save file. ([legacy] If omitted, the combination of `path` and `step` will be used instead.)
        fields_names: [legacy] grid's fields' names.
        path: data's parent directory
        step: [legacy] step number to load

    Warning:
        The legacy support (ordered and unordered ``.npz``) may be deprecated in the future, and only exists for loading old results.

    Todo:
        Once we deprecate the legacy support (3.x ?), remove the additional obsolete arguments, and reflect in usages

    Returns:
        a dict containing the saved data, ``{fields: the grid's fields, t: step time, N_points: grid size, k_min: grid k min, elapsed_time: real elapsed time, k0: grid.k0}``
    """

    @load_hdf5
    def load_step_hdf5(f: h5py.File):
        """load for hdf5"""
        group = f[f"/step_{step}"]
        fields = {name: group[name][:] for name in group}
        step_data = {**dict(group.attrs), "fields": fields}
        return step_data

    try:  # hdf5 save
        return load_step_hdf5(path, filepath)
    except FileNotFoundError:  # Legacy support: .npz saves
        if filepath is None:
            filepath = f"{path}/fields_{step}.npz"
        data = np.load(filepath, allow_pickle=True)

        try:
            fields = {name: data[f"field_{name}"] for name in fields_names}
            step_data = {
                "fields": fields,
                "t": data["t"],
                "N_points": data["N_points"],
                "k_min": data["k_min"],
                "elapsed_time": data["elapsed_time"],
                "k0": data["k0"],
            }
        except KeyError:  # Legacy support for old npz save /!\ the order of `fields_names` is important
            reserved_arrays = 5
            fields = {fields_names[i]: data[f"arr_{reserved_arrays + i}"] for i in range(len(fields_names))}
            step_data = {
                "fields": fields,
                "t": data["arr_0"],
                "N_points": data["arr_1"],
                "k_min": data["arr_2"],
                "elapsed_time": data["arr_3"],
                "k0": data["arr_4"],
            }

        return step_data


@store_hdf5
def save_step(path: str, step: int, grid: Grid, t: float, dt: float, time_tracker: TimeTracker) -> None:
    """Save current step's fields to a .h5 file ``save.h5`` (appending if existing), as a new dataset ``step_<step>``

    Args:
        path: folder to save to
        grid: grid containing the fields to save
        t: time of the step
        dt: timestep used
        step: # of the step
        time_tracker: object that tracks elapsed time
    """
    path: h5py.File  # from the decorator
    group_name = f"step_{step}"
    group = path.create_group(group_name)
    group.attrs.create("t", t)
    group.attrs.create("N_points", grid.N_points)
    group.attrs.create("k_min", grid.k_min)
    group.attrs.create("k0", grid.k0)
    group.attrs.create("dt", dt)
    for k, v in time_tracker.elapsed_time.items():
        group.attrs.create(f"ttrack_{k}", v)
    for name in grid.fields_names:
        group.create_dataset(name, data=grid.fields[name])


# DataExplorer drawables
def add_drawables(path: str, drawables: dict) -> None:
    """Adds a drawables data dict to a DataExplorer result

    Args:
        path: the path where the DataExplorer result is
        drawables: the dict to save
    """
    # noinspection PyTypeChecker
    np.save(f"{path}/drawables", {"drawables": drawables})  # saved as an np.ndarray
