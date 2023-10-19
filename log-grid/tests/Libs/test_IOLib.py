"""utests for IOLib.py"""
import os.path

from pyloggrid.Libs.singlethread_numpy import np

np.zeros(0)  # Do not remove - used to force singlethread_numpy to be imported before the other libraries when refactoring

from typing import Optional

import deepdiff
import h5py
import orjson

from pyloggrid.Libs.datasci import randcomplex_like
from pyloggrid.Libs.IOLib import add_drawables, load_step, read_json_settings, save_step, update_json_settings
from pyloggrid.Libs.misc import TimeTracker
from pyloggrid.LogGrid.Grid import Grid


def test_update_parameters(tmp_path):
    path = str(tmp_path)
    sett_path = f"{path}/settings.json"
    sett_bk_path = f"{path}/settings.json_bk"
    sett_new_path = f"{path}/settings.json_new"

    # create new settings
    params = {"a": "aa", 1: [3, 4, 5], 2: None, 3: np.nan}
    update_json_settings(path, params=params)
    assert os.path.isfile(sett_path)
    assert not os.path.isfile(sett_new_path)
    assert not os.path.isfile(sett_bk_path)
    with open(sett_path, "r") as f:
        data = orjson.loads(f.read())
    assert len(deepdiff.DeepDiff(data, {"a": "aa", "1": [3, 4, 5], "2": None, "3": None})) == 0

    # update existing settings
    params2 = {"a": "bb", 1: 335}
    update_json_settings(path, params=params2, update=True)
    with open(sett_path, "r") as f:
        data = orjson.loads(f.read())
    assert len(deepdiff.DeepDiff(data, {"a": "bb", "1": 335, "2": None, "3": None})) == 0
    assert os.path.isfile(sett_bk_path)
    with open(sett_bk_path, "r") as f:
        data = orjson.loads(f.read())
    assert len(deepdiff.DeepDiff(data, {"a": "aa", "1": [3, 4, 5], "2": None, "3": None})) == 0

    # overwrite existing settings
    params3 = {"a": "cc", 8: 111}
    update_json_settings(path, params=params3)
    with open(sett_path, "r") as f:
        data = orjson.loads(f.read())
    assert len(deepdiff.DeepDiff(data, {"a": "cc", "8": 111})) == 0
    with open(sett_bk_path, "r") as f:
        data = orjson.loads(f.read())
    assert len(deepdiff.DeepDiff(data, {"a": "bb", "1": 335, "2": None, "3": None})) == 0


def test_read_json_settings(tmp_path):
    path = str(tmp_path)
    sett_bk_path = f"{path}/settings.json_bk"
    sett_new_path = f"{path}/settings.json_new"

    # test reading json backup
    params = {"a": "aa", "1": [3, 4, 5], "2": None, "3": None}
    with open(sett_bk_path, "wb") as f:
        f.write(orjson.dumps(params, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY))
    assert type(read_json_settings(path)) is dict
    assert len(deepdiff.DeepDiff(read_json_settings(path), params)) == 0
    sett_path = f"{path}/settings.json"
    assert not os.path.isfile(sett_path)
    assert not os.path.isfile(sett_new_path)

    # test reading main json
    params2 = {"a": "bb", "1": 222}
    with open(sett_path, "wb") as f:
        f.write(orjson.dumps(params2, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY))
    assert type(read_json_settings(path)) is dict
    assert len(deepdiff.DeepDiff(read_json_settings(path), params2)) == 0
    assert not os.path.isfile(sett_new_path)

    # test reading main json if no backup exists
    os.remove(sett_bk_path)
    assert type(read_json_settings(path)) is dict
    assert len(deepdiff.DeepDiff(read_json_settings(path), params2)) == 0


def test_save_step(tmp_path):
    path = str(tmp_path)

    def save_step_D(D: int) -> None:
        """test save_step with dimension D"""
        step = 50
        t = 0.12325
        dt = t / 100
        N_points = 20
        k0 = False
        k_min = 55.5
        grid = Grid(D=D, l_params={"a": None, "b": None, "plastic": False}, N_points=N_points, k_min=k_min, fields_name=["f1", "f2"], k0=k0)
        grid.fields["f1"] = randcomplex_like(grid.fields["f1"])
        grid.fields["f2"] = randcomplex_like(grid.fields["f1"])
        elapsed_time = 15645.2
        save_step(path, step=step, grid=grid, t=t, dt=dt, time_tracker=TimeTracker({"total": elapsed_time}))

        fpath = os.path.join(path, "output.h5")
        assert os.path.isfile(fpath)
        with h5py.File(fpath, "r") as f:
            assert f"step_{step}" in f
            data = f[f"step_{step}"]
            data_attrs = data.attrs

            assert t == data_attrs["t"]
            assert dt == data_attrs["dt"]
            assert N_points == data_attrs["N_points"]
            assert k_min == data_attrs["k_min"]
            assert elapsed_time == data_attrs["ttrack_total"]
            assert k0 == data_attrs["k0"]
            assert np.array_equal(grid.fields["f1"], data["f1"])
            assert np.array_equal(grid.fields["f2"], data["f2"])

        os.remove(fpath)
        os.remove(f"{fpath[:-3]}_bk.h5")

    save_step_D(1)
    save_step_D(2)
    save_step_D(3)


def test_load_step(tmp_path):
    path = str(tmp_path)

    def load_step_D(D: int, filepath: Optional[bool] = None) -> None:
        """test load_step with dimension D"""
        step = 22
        k0 = True
        t = 0.123258
        dt = t / 100
        N_points = 21
        k_min = 55.52
        fields_name = ["f1", "f2"]
        grid = Grid(D=D, l_params={"a": 1, "b": 2, "plastic": False}, N_points=N_points, k_min=k_min, fields_name=fields_name, k0=k0)
        grid.fields["f1"] = randcomplex_like(grid.fields["f1"])
        grid.fields["f2"] = randcomplex_like(grid.fields["f1"])
        elapsed_time = 15645.2
        save_step(path, step=step, grid=grid, t=t, dt=dt, time_tracker=TimeTracker({"total": elapsed_time}))

        for corrupted in [0, 1, 2]:  # 1 = missig, 2 = corrupted
            fpath = os.path.join(path, "output.h5") if filepath else None
            if corrupted and not fpath:
                if corrupted == 1:
                    os.remove(os.path.join(path, "output.h5"))
                else:
                    with open(os.path.join(path, "output.h5"), "w") as f:
                        f.write("corrupted")
            step_data = load_step(fields_names=fields_name, path=path, step=step, filepath=fpath)

            assert t == step_data["t"]
            assert dt == step_data["dt"]
            assert N_points == step_data["N_points"]
            assert k_min == step_data["k_min"]
            assert elapsed_time == step_data["ttrack_total"]
            assert k0 == step_data["k0"]
            assert np.array_equal(grid.fields["f1"], step_data["fields"]["f1"])
            assert np.array_equal(grid.fields["f2"], step_data["fields"]["f2"])

        os.remove(f"{path}/output.h5")
        os.remove(f"{path}/output_bk.h5")

    # load with path and step
    load_step_D(1)
    load_step_D(2)
    load_step_D(3)

    # load with filepath
    load_step_D(1, True)
    load_step_D(2, True)
    load_step_D(3, True)


def test_add_drawables(tmp_path):
    # Create sample data
    drawables = {"a": [1, 2, 3], "b": [4, 5, 6]}

    # Call function
    # noinspection PyTypeChecker
    add_drawables(tmp_path, drawables)

    # Load saved file
    filepath = os.path.join(tmp_path, "drawables.npy")
    saved_data = np.load(filepath, allow_pickle=True).item()

    # Check if saved data matches input data
    assert saved_data == {"drawables": drawables}
