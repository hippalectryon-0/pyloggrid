"""
Treat 3D results
Specialized for NS3D
"""

from pyloggrid.Libs.singlethread_numpy import np

np.zeros(0)  # Do not remove - used to force singlethread_numpy to be imported before the other libraries when refactoring

import logging

from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

from pyloggrid.Libs import datasci
from pyloggrid.Libs.custom_logger import setup_custom_logger
from pyloggrid.Libs.plotLib import interactive_3D_logplot_by_z, interactive_spectrum, labels, pltshowm, scatter
from pyloggrid.LogGrid.DataExplorer import DataExplorer, PlotFun
from pyloggrid.LogGrid.Grid import Grid

logger = setup_custom_logger(__name__, level=logging.INFO)


## Getters and plotters


# Simulation parameters: time, grid size (N)
def get_simu_params(grid: Grid, _t: float, _simu_params: dict) -> dict:
    """time, number of grid points"""
    return {"N": grid.N_points}


def plot_simu_params(drawables: PlotFun) -> None:
    """time / speed, number of grid points"""
    t, N = drawables("t", "N")

    fig, axs = plt.subplot_mosaic("A;B", sharex=True)
    ax1, ax2 = axs["A"], axs["B"]
    ax1_2 = ax1.twinx()

    (l1,) = scatter(ax1, np.gradient(t), color="orange", label="speed (left, t/it)")
    (l2,) = scatter(ax1_2, t, label="time (right)", linestyle="-")
    ax1.set_ylabel("speed")
    ax1_2.set_ylabel("t")
    ax1_2.legend(handles=[l1, l2])
    ax1.set_title("Time/speed of the simulation vs. iteration")

    scatter(ax2, N, color="red")
    ax2.set_ylabel("Grid size (N)")
    ax2.set_xlabel("Simulation step")
    pltshowm(legend=False, save=f"{save_path}/simu_params.png")


# Spectrum and Energy
def get_spectrum_and_energy(grid: Grid, _t: float, _simu_params: dict) -> dict:
    """
    LHS: spectrum vs ks
    RHS: energy vs time
    slider: time
    """

    def spectrum_kinetic(fields: dict, k: np.ndarray) -> float:
        """Kinetic energy"""
        ux, uy, uz = fields["ux"], fields["uy"], fields["uz"]
        return np.real(ux[k] * np.conj(ux[k]) + uy[k] * np.conj(uy[k]) + uz[k] * np.conj(uz[k]))

    # noinspection PyMissingOrEmptyDocstring
    def spectrum_kinetic_x(fields: dict, k: np.ndarray) -> float:
        ux = fields["ux"]
        return np.real(ux[k] * np.conj(ux[k]))

    # noinspection PyMissingOrEmptyDocstring
    def spectrum_kinetic_y(fields: dict, k: np.ndarray) -> float:
        uy = fields["uy"]
        return np.real(uy[k] * np.conj(uy[k]))

    # noinspection PyMissingOrEmptyDocstring
    def spectrum_kinetic_z(fields: dict, k: np.ndarray) -> float:
        uz = fields["uz"]
        return np.real(uz[k] * np.conj(uz[k]))

    E_k = grid.physics.spectrum(spectrum_kinetic)
    E_kx = grid.physics.spectrum(spectrum_kinetic_x)
    E_ky = grid.physics.spectrum(spectrum_kinetic_y)
    E_kz = grid.physics.spectrum(spectrum_kinetic_z)

    return {"E_k": E_k, "E_kx": E_kx, "E_ky": E_ky, "E_kz": E_kz, "E": grid.physics.energy(), "ks": grid.ks_1D}


def plot_spectrum_and_energy(drawables: PlotFun) -> None:
    """Plot spectra and energy"""
    ts, E_k, E_kx, E_ky, E_kz, E, ks = drawables("t", "E_k", "E_kx", "E_ky", "E_kz", "E", "ks")
    _ = interactive_spectrum(ts, ks, {"$E_k$": E_k, "$E_{kx}$": E_kx, "$E_{ky}$": E_ky, "$E_{kz}$": E_kz}, {"$E$": E})
    pltshowm(legend=False, compact=False)


def get_spectrum(dexp: DataExplorer) -> dict:
    """energy spectrum"""
    step = dexp.N_steps  # last step
    N_avg = 25

    # average spectra over last N_avg steps
    E_ks = []
    ks = np.zeros(0)
    for curr_step in range(max(step - N_avg, 1), step + 1):
        _, grid = dexp.load_step(curr_step)
        if grid.ks_1D.size > ks.size:
            ks = grid.ks_1D

        def spectrum_kinetic(fields: dict, k: np.ndarray) -> float:
            """Kinetic energy"""
            ux, uy, uz = fields["ux"], fields["uy"], fields["uz"]
            return np.real(ux[k] * np.conj(ux[k]) + uy[k] * np.conj(uy[k]) + uz[k] * np.conj(uz[k]))

        E_ks.append(grid.physics.spectrum(spectrum_kinetic))

    E_ks = datasci.ragged_array_to_array(E_ks)
    E_k = datasci.logmean(E_ks, axis=0)

    return {"ks": ks, "E_k": E_ks[-1], "E_k_avg": E_k}


def plot_spectrum(drawables: PlotFun) -> None:
    """energy spectrum"""
    ks, E_k, E_k_avg = drawables("ks", "E_k", "E_k_avg")

    # range over which to perform the fit
    mask = (ks > 1) & (ks < 100)
    a, xi = datasci.powerlaw_fit(ks, E_k_avg, mask=mask)
    plt.loglog(ks[mask], a * ks[mask] ** xi, "--", label=f"fit $ak^\\xi$, $\\xi$={xi:.3}", linewidth=5)

    plt.loglog(ks, E_k, label="$u^2$", linewidth=10)
    plt.loglog(ks, E_k_avg, label="$u^2 avg$")
    labels("$k$", "$E(k)$", "Energy spectrum")
    pltshowm(save=f"{save_path}/spectrum.png")


# Anisotropy
def get_bxyz(grid: Grid, _t: float, _simu_params: dict) -> dict:
    """velocity anisotropy"""
    ux, uy, uz = grid.field("ux", "uy", "uz")
    # mask = grid.ks_modulus > grid.k_min * grid.l ** 4
    # ux, uy, uz = ux[mask], uy[mask], uz[mask]
    E2 = 2 * grid.physics.energy()
    bzz = 1 / 3 - grid.maths.self_inner_product(uz) / E2
    bxx = 1 / 3 - grid.maths.self_inner_product(ux) / E2
    byy = 1 / 3 - grid.maths.self_inner_product(uy) / E2

    return {"bzz": bzz, "bxx": bxx, "byy": byy}


def plot_bxyz(drawables: PlotFun) -> None:
    """velocity anisotropy"""
    ts, bzz, bxx, byy = drawables("t", "bzz", "bxx", "byy")
    N = min(200, bzz.size // 3)  # width of smoothing

    plt.plot(ts, bzz, label="$b_{zz}$")
    plt.plot(ts, bxx, label="$b_{xx}$")
    plt.plot(ts, byy, label="$b_{yy}$")
    plt.plot(ts, savgol_filter(bzz, N, 3), "--", label="$b_{zz}$")
    plt.plot(ts, savgol_filter(bxx, N, 3), "--", label="$b_{xx}$")
    plt.plot(ts, savgol_filter(byy, N, 3), "--", label="$b_{yy}$")
    plt.axhline(y=1 / 3, linestyle="--", color="grey")
    plt.axhline(y=-2 / 3, linestyle="--", color="grey")
    labels("$t$", r"$b_{ii} = \frac{1}{3}-\frac{u_i^2}{u^2}$", "Velocity anisotropy")
    pltshowm(save=f"{save_path}/bxyz.png")


# 3D vis
def get_3D(dexp: DataExplorer) -> dict:
    """3D plots"""
    ts, uzs = [], []
    ks, ks_mod = None, None

    max_N = 0  # get max size
    for curr_step in range(1, dexp.N_steps + 1):
        data: dict = dexp.load_step(curr_step, grid=False)
        max_N = max(max_N, data["N_points"])

    for curr_step in range(1, dexp.N_steps + 1):
        data = dexp.load_step(curr_step, grid=False)
        fields, k_min = data["fields"], data["k_min"]
        grid = dexp.getgrid(max_N, k_min, fields)  # load as max size

        uzs.append(grid.field("uz"))
        ts.append(data["t"])
        if ks is None:
            ks, ks_mod = grid.ks, grid.ks_modulus

    return {"t": np.array(ts), "uz": np.array(uzs), "ks": ks, "ks_mod": ks_mod}


def plot_3D(drawables: PlotFun) -> None:
    """Plot 3D fields"""
    ts, uz, ks, ks_mod = drawables("t", "uz", "ks", "ks_mod")
    kx, ky, kz = ks
    ax, _ = interactive_3D_logplot_by_z(kx, ky, uz, kz[0, 0])
    ax.set_xlabel("$kx$")
    ax.set_ylabel("$ky$")
    ax.set_zlabel("$uz$")
    pltshowm(legend=False, compact=False)


# epsilon_inj, epsilon_diss
def get_epsilon(grid: Grid, _t: float, simu_params: dict) -> dict:
    r"""energy injection and dissipation
    /!\ the forcing is hardcoded, and needs to be updated if changed in the simulation !"""
    Re_F = simu_params["Re_F"]
    if Re_F is None:
        Re_F = np.inf

    # noinspection PyMissingOrEmptyDocstring
    def get_forcing(grid: Grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        f_rot = datasci.randcomplex_seeded_by_array(grid.ks, 1337)
        f = grid.maths.rot3D_inv(f_rot)
        f[:, grid.ks_modulus > grid.k_min * grid.l**3] = 0
        f = f / np.sqrt(grid.maths.self_inner_product(f))
        assert np.isclose(grid.maths.self_inner_product(np.array(f)), 1)  # check <f,f>=1
        f = f * f0
        assert np.isclose(np.max(np.abs(grid.maths.div3D(f))), 0)  # check div=0

        fx, fy, fz = f
        return fx, fy, fz

    ux, uy, uz = grid.field("ux", "uy", "uz")
    u = np.array([ux, uy, uz])
    f = get_forcing(grid)
    visc = grid.maths.laplacian * np.sqrt(f0) * grid.L**1.5 / Re_F

    eps_inj = grid.maths.inner_product(f, u)
    eps_diss = grid.maths.inner_product(u, visc * u)
    E = grid.physics.energy()

    return {"epsilon_inj": eps_inj, "epsilon_diss": eps_diss, "E": E}


def plot_epsilon(drawables: PlotFun):
    r"""energy injection and dissipation
    /!\ discrepancy between dtE and ei+e_d may just be due to the computation of dtE when the timestep is not constant"""
    eps_inj, ts, eps_diss, E = drawables("epsilon_inj", "t", "epsilon_diss", "E")
    scatter(None, ts, eps_inj, label=r"$\epsilon_{inj}$")
    scatter(None, ts, -eps_diss, label=r"$-\epsilon_{diss}$")
    scatter(None, ts, np.gradient(E, edge_order=2) / np.gradient(ts, edge_order=2), "x", label=r"$\partial_t E$")
    scatter(None, ts, eps_inj + eps_diss, "+", label=r"$\epsilon_{inj}+\epsilon_{diss}$")
    labels("$t$", "", "energy injection and dissipation")
    plt.yscale("asinh")
    pltshowm(save=f"{save_path}/epsilon.png")


# RMS
def get_uRMS(dexp: DataExplorer) -> dict:
    """RMS u"""
    u2, ts, Vs = [], [], []
    for curr_step in range(1, dexp.N_steps + 1):
        data, grid = dexp.load_step(curr_step)
        t = data["t"]
        u2.append(2 * grid.physics.energy())
        ts.append(t)
        Vs.append(grid.V)

    u2, Vs, ts = np.array(u2), np.array(Vs), np.array(ts)
    uRMS_perstep = np.sqrt(u2) / Vs

    uRMS = np.sqrt(np.sum(u2 * np.gradient(ts)) / (ts[-1] - ts[0]))

    return {"uRMS_perstep": uRMS_perstep, "t": ts, "uRMS": uRMS}


def plot_uRMS(drawables: PlotFun):
    """RMS u"""
    ts, uRMS, uRMS_perstep = drawables("t", "uRMS", "uRMS_perstep")
    plt.semilogy(ts, uRMS_perstep, "o", label=r"$u^{RMS}=\sqrt{\sum u^2}/V$")
    plt.axhline(y=uRMS, linestyle="--", label=r"$u^{RMS}_{total}=\sqrt{1/T\int\sum u^2/V}$", color="black")
    labels("$t$", "", "RMS vs $t$")
    pltshowm(save=f"{save_path}/uRMS.png")


draw_funcs = {
    # "simu_params": {"get": get_simu_params, "plot": plot_simu_params},
    "spectrum_and_energy": {"get": get_spectrum_and_energy, "plot": plot_spectrum_and_energy},
    # "spectrum": {"get": get_spectrum, "plot": plot_spectrum, "perframe": False},
    # "bxyz": {"get": get_bxyz, "plot": plot_bxyz},
    # "3D": {"get": get_3D, "plot": plot_3D, "perframe": False},
    "epsilon": {"get": get_epsilon, "plot": plot_epsilon},
    # "uRMS": {"get": get_uRMS, "plot": plot_uRMS, "perframe": False},
}
f0 = 1
Re_F = 1e3

save_path = f"results/save_3D_f0{f0:.2e}_ReF{Re_F:.2e}"  # where the simulation data is saved
temp_path = "results/temp"  # temp path to store images before compiling to video
N_points = 500  # how many time points max to load
n_jobs = 3  # parallelisation
loadfromsave = False  # load already computed results /!\ if your data has changed, this may show the old data

## Do not edit below unless you know what you're doing

dexp = DataExplorer(save_path)
dexp.display(draw_funcs=draw_funcs, N_points=N_points, n_jobs=n_jobs, loadfromsave=loadfromsave)
