"""
3D Navier Stokes
"""

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from pyloggrid.Libs.singlethread_numpy import np

np.zeros(0)  # Do not remove - used to force singlethread_numpy to be imported before the other libraries when refactoring

from pyloggrid.Libs.custom_logger import setup_custom_logger
from pyloggrid.Libs.datasci import randcomplex_like, randcomplex_seeded_by_array
from pyloggrid.LogGrid.Framework import Grid, Solver

logger = setup_custom_logger(__name__)

f0, fx, fy, fz = 1, None, None, None


def get_forcing(grid: Grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3D pseudorandom forcing, cached"""
    global fx, fy, fz
    if fx is None:
        f_rot = randcomplex_seeded_by_array(grid.ks, 1337)
        f = grid.maths.rot3D_inv(f_rot)
        f[:, grid.ks_modulus > grid.k_min * grid.l**3] = 0
        f = f / np.sqrt(grid.maths.self_inner_product(f))
        assert np.isclose(grid.maths.self_inner_product(np.array(f)), 1)  # check <f,f>=1
        f = f * f0
        assert np.isclose(np.max(np.abs(grid.maths.div3D(f))), 0)  # check div=0

        fx, fy, fz = f
    return fx, fy, fz


# noinspection PyMissingOrEmptyDocstring
def equation_nonlinear(_t: float, grid: Grid, _simu_params: dict) -> dict[str, np.ndarray]:
    M = grid.maths

    ux, uy, uz = grid.field("ux", "uy", "uz")

    # Evolution
    uxdxux, uydyux, uxdxuy, uydyuy, uzdzux, uzdzuy, uxdxuz, uydyuz, uzdzuz = M.convolve_batch(
        (
            (ux, M.dx * ux),
            (uy, M.dy * ux),
            (ux, M.dx * uy),
            (uy, M.dy * uy),
            (uz, M.dz * ux),
            (uz, M.dz * uy),
            (ux, M.dx * uz),
            (uy, M.dy * uz),
            (uz, M.dz * uz),
        )
    )

    # Forcing
    fx, fy, fz = get_forcing(grid)

    # Evolution w/o pressure
    dux_dt = -uxdxux - uydyux - uzdzux + fx
    duy_dt = -uxdxuy - uydyuy - uzdzuy + fy
    duz_dt = -uxdxuz - uydyuz - uzdzuz + fz

    # Add pressure
    dux_dt, duy_dt, duz_dt = grid.maths.P_projector([dux_dt, duy_dt, duz_dt])

    return {"ux": dux_dt, "uy": duy_dt, "uz": duz_dt}


# noinspection PyMissingOrEmptyDocstring
def equation_linear(_t: float, grid: Grid, simu_params: dict) -> dict[str, np.ndarray]:
    Re_F = simu_params["Re_F"]
    nu = np.sqrt(f0) * grid.L**1.5 / Re_F

    visc = grid.maths.laplacian * nu

    return {"ux": visc, "uy": visc, "uz": visc}


# noinspection PyMissingOrEmptyDocstring
def initial_conditions(fields: dict[str, np.ndarray], grid: Grid, _simu_params: dict) -> dict[str, np.ndarray]:
    grid = grid.to_new_size_empty(N_points)
    u = get_forcing(grid)
    randu = grid.maths.rot3D_inv(randcomplex_like(np.array(u)))
    u = u + randu * 1e-200  # make sure ~all the modes are nonzero
    ux, uy, uz = u

    fields["ux"] = ux
    fields["uy"] = uy
    fields["uz"] = uz

    return fields


def update_gridsize(grid: Grid) -> int | None:
    """update the grid size based on the fraction of energy contained in the outermost layers"""
    global fx  # if grid updated, reset f so that it's recomputed
    E = grid.physics.energy()
    ux, uy, uz = grid.field("ux", "uy", "uz")
    mask = grid.ks_modulus > grid.k_min * grid.l ** (grid.N_points - 1)
    comp = np.max(np.abs(ux[mask]) + np.abs(uy[mask]) + np.abs(uz[mask]))
    if comp / np.sqrt(E) > 1e-100:
        fx = None
        return grid.N_points + 1
    if comp / np.sqrt(E) < 1e-170 and grid.N_points > 5:
        fx = None
        return grid.N_points - 1


fields = ["ux", "uy", "uz"]  # the scalar fields to simulate
D = 3  # the dimension of the space
l_params = {"plastic": False, "a": 1, "b": 2}  # the grid spacing's parameters
Re_F = 1e3
simu_params = {"Re_F": Re_F}  # scalar parameter of the simulation, passed to the equation

rtol = 1e-4  # relative tolerance of the solver
n_threads_convolution = 4  # parallelization
N_points = 6  # initial size of the grid

save_path = f"results/save_3D_f0{f0:.2e}_ReF{Re_F:.2e}"  # save path
end_simulation = {"t": 2000, "ode_step": 1e10}  # when to end the simulation
save_one_in = 50  # save one step every N real steps

# Do not edit below unless you know what you're doing #

logger.info(f"Chosen parameters {simu_params}")

solver = Solver(
    fields_names=fields,
    equation_nl=equation_nonlinear,
    equation_l=equation_linear,
    D=D,
    l_params=l_params,
    simu_params=simu_params,
    n_threads=n_threads_convolution,
)
solver.solve(
    solver_params={"rtol": rtol},
    save_path=save_path,
    save_one_in=save_one_in,
    end_simulation=end_simulation,
    update_gridsize_cb=update_gridsize,
    initial_conditions=initial_conditions,
)
