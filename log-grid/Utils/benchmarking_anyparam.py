"""
Benchmark performance of (3)D simulation vs. arbitrary parameter, incl. # of threads
Plot results
"""

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from pyloggrid.Libs.singlethread_numpy import np

np.zeros(0)
import time
from typing import Any

from matplotlib import pyplot as plt

from pyloggrid.Libs.custom_logger import setup_custom_logger
from pyloggrid.Libs.plotLib import labels, pltshowm
from pyloggrid.LogGrid.Framework import Grid

plt.rcParams.update({"font.size": 20, "lines.linewidth": 5, "lines.markersize": 15})
logger = setup_custom_logger(__name__)


def do_cycle(params: dict) -> None:
    """the part to benchmark"""
    grid, f, g = params["grid"], params["f"], params["g"]
    grid.maths.convolve(f, g)


def init_cycle(params: dict) -> dict:
    """called before starting a cycle"""
    grid = params["grid"]
    params["f"] = np.random.randn(*grid.ks_modulus.shape).astype("complex")
    params["g"] = np.random.randn(*grid.ks_modulus.shape).astype("complex")
    return params


# This is where you set which parameter is benchmarked
def init_newparam(x: Any) -> dict:
    """called after changing the param's value"""
    l_params = {"plastic": False, "a": 1, "b": 2}
    n_threads = 1

    grid = Grid(D, l_params, x, n_threads=n_threads, fields_name=[], k0=k0)

    return {"grid": grid}


## Settings
x = np.linspace(1, 20, 10).astype(int)  # varying parameter
D = 3
k0 = False
x_name = "Grid size"
x_title = rf"Time for one convolution vs grid size, D={D}, k0={k0} for $\lambda=\phi$"
N_cycles = 30  # How many runs to average

# Do the benchmark
x_times = []
for v in x:
    logger.info(f"Testing param={v}")
    params = init_newparam(v)
    cycles_times = []
    for _ in range(N_cycles):
        params = init_cycle(params)
        start = time.perf_counter()
        do_cycle(params)
        cycles_times.append(time.perf_counter() - start)
    x_times.append(cycles_times)
x_times_mean = np.mean(x_times, axis=1)

logger.info(f"Benchmark times: {', '.join(f'{v}: {t:.2e}s' for v, t in zip(x, x_times_mean))}")
# Plot results
plt.subplot(1, 2, 1)
plt.plot(x, np.max(x_times_mean) / x_times_mean, "o")
#     plt.plot(n_threads_convolution, x_times[0, 0] / x_times[i], 'o', color=color, label=f"n_parallel={n_parallel}")
# plt.ylim(top=1.1 * x_times[0, 0] / np.min(x_times))
labels(x_name, "speed=1/time, rescaled", "speed")
plt.subplot(1, 2, 2)
plt.semilogy(x, x_times_mean, "o")
for i in range(len(x)):
    plt.scatter([x[i]] * N_cycles, x_times[i], marker="o", color="black", s=50, alpha=3 / N_cycles, zorder=99)
labels(x_name, "Mean time (s)", "duration")
plt.suptitle(f"Benchmark for {x_name}\n{x_title}")
pltshowm(legend=False, save="benchmarking_anyparam.png", full=True)
