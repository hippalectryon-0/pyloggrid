"""3D HRB simulations"""

from pyloggrid.Libs.singlethread_numpy import np

np.zeros(0)  # Do not remove - used to force singlethread_numpy to be imported before the other libraries when refactoring

from pyloggrid.Libs.custom_logger import setup_custom_logger
from pyloggrid.Libs.datasci import randcomplex_like
from pyloggrid.LogGrid.Framework import Grid, Solver

logger = setup_custom_logger(__name__)


# noinspection PyMissingOrEmptyDocstring
def equation_nonlinear(_t: float, grid: Grid, _simu_params: dict) -> dict[str, np.ndarray]:
    M = grid.maths
    ux, uy, uz, theta = grid.field("ux", "uy", "uz", "theta")

    # Evolution
    uxdxux, uydyux, uzdzux, uxdxuy, uydyuy, uzdzuy, uxdxuz, uydyuz, uzdzuz, uxdxtheta, uydytheta, uzdztheta = M.convolve_batch(
        (
            (ux, M.dx * ux),
            (uy, M.dy * ux),
            (uz, M.dz * ux),
            (ux, M.dx * uy),
            (uy, M.dy * uy),
            (uz, M.dz * uy),
            (ux, M.dx * uz),
            (uy, M.dy * uz),
            (uz, M.dz * uz),
            (ux, M.dx * theta),
            (uy, M.dy * theta),
            (uz, M.dz * theta),
        )
    )

    # w/o pressure
    dux_dt = -uxdxux - uydyux - uzdzux
    duy_dt = -uxdxuy - uydyuy - uzdzuy
    duz_dt = -uxdxuz - uydyuz - uzdzuz + theta
    dtheta_dt = -uxdxtheta - uydytheta - uzdztheta + uz

    # Add pressure
    dux_dt, duy_dt, duz_dt = grid.maths.P_projector([dux_dt, duy_dt, duz_dt])

    return {"ux": dux_dt, "uy": duy_dt, "uz": duz_dt, "theta": dtheta_dt}


# noinspection PyMissingOrEmptyDocstring
def equation_linear(_t: float, grid: Grid, simu_params: dict) -> dict[str, np.ndarray]:
    M = grid.maths
    Pr, Ra = simu_params["Pr"], simu_params["Ra"]

    f = simu_params["f"] * np.ones_like(grid.ks_modulus)
    f[grid.ks_modulus > grid.k_min * grid.l ** 3] = 0

    visc = np.sqrt(Pr / Ra) * M.laplacian - f
    visc_theta = 1 / np.sqrt(Ra * Pr) * M.laplacian - f

    return {"ux": visc, "uy": visc, "uz": visc, "theta": visc_theta}


# noinspection PyMissingOrEmptyDocstring
def initial_conditions(fields: dict[str, np.ndarray], grid: Grid, _simu_params: dict) -> dict[str, np.ndarray]:
    grid = grid.to_new_size_empty(N_points)
    ks = grid.ks_modulus

    wx = np.zeros_like(ks, dtype="complex")
    wy = np.zeros_like(ks, dtype="complex")
    wz = np.zeros_like(ks, dtype="complex")
    theta = randcomplex_like(ks) * np.ones_like(ks, dtype="complex") / 100
    theta[ks > grid.k_min * 4] = 0
    ux, uy, uz = grid.maths.rot3D_inv([wx, wy, wz])

    fields["theta"] = theta
    fields["ux"] = ux
    fields["uy"] = uy
    fields["uz"] = uz

    return fields


def update_gridsize(grid: Grid) -> int | None:
    """update the grid size based on the fraction of energy contained in the outermost layers"""
    E = grid.physics.energy()
    ux, uy, uz, theta = grid.field("ux", "uy", "uz", "theta")
    mask = grid.ks_modulus > grid.k_min * grid.l ** (grid.N_points - 1)
    # grid
    comp = np.max(np.abs(ux[mask]) + np.abs(uy[mask]) + np.abs(uz[mask]) + np.abs(theta[mask]))
    if comp / np.sqrt(E) > 1e-100:
        return grid.N_points + 1
    if comp / np.sqrt(E) < 1e-170 and grid.N_points > 5:
        return grid.N_points - 1


Pr = 1
Ra = 1e12
print(f"Chosen parameters Pr={Pr}, Ra={Ra}")
fields = ["ux", "uy", "uz", "theta"]
D = 3
l_params = {"plastic": False, "a": 1, "b": 2}
n_threads_convolution = 6
simu_params = {"Pr": Pr, "Ra": Ra, "f": 1}
N_points = 13
rtol = 1e-1

save_path = "results/saveRB"
loadfromsave = False

## Do not edit below ##
solver = Solver(
    fields_names=fields,
    equation_nl=equation_nonlinear,
    D=D,
    equation_l=equation_linear,
    l_params=l_params,
    simu_params=simu_params,
    n_threads=n_threads_convolution,
)
solver.solve(
    save_path=save_path,
    initial_conditions=initial_conditions,
    save_one_in=100,
    update_gridsize_cb=update_gridsize,
    solver_params={"rtol": rtol, "adapt_cutoff": 1e-5},
)
