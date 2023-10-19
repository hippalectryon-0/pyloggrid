"""
Benchmark multithreading performance of (3)D simulation vs. number of CPUs
Plot results, best time vs therotical ideal time (time on 1 CPU / # of CPUs)
"""
import os
import sys

sys.path.insert(1, os.path.join(os.getcwd(), ".."))
from pyloggrid.Libs.singlethread_numpy import np

np.zeros(0)
import math
import time

import matplotlib
from matplotlib import pyplot as plt

from pyloggrid.Libs.custom_logger import setup_custom_logger
from pyloggrid.Libs.plotLib import pltshowm
from pyloggrid.LogGrid.Framework import Grid

plt.rcParams.update({"font.size": 20, "lines.linewidth": 5, "lines.markersize": 15})
logger = setup_custom_logger(__name__)

## Parameters

D = 3  # Dimension to benchmark
N_points = 15  # Grid size
l_params = {"plastic": False, "a": 1, "b": 2}
n_threads_convolution = np.array([1, 2, 3, 4, 5, 8, 12, 16])
n_parallel_convolution = np.array([1, 2, 3, 4, 5, 8, 12, 16])
k0 = False
number_cycles = 1000  # Average each point over this many convolutions


## Do not edit below ##
def benchmark_function(grid: Grid, n_batch: int) -> float:
    """
    The function to benchmark. In our case, one convolution.
    :return: the time elapsed
    """

    if n_batch > 1:
        fgs = [(np.random.randn(*grid.ks_modulus.shape).astype("complex"), np.random.randn(*grid.ks_modulus.shape).astype("complex")) for _ in range(n_batch)]

        t0 = time.perf_counter()
        grid.maths.convolve_batch(fgs)

    else:
        f, g = np.random.randn(*grid.ks_modulus.shape).astype("complex"), np.random.randn(*grid.ks_modulus.shape).astype("complex")
        t0 = time.perf_counter()
        grid.maths.convolve(f, g)

    return time.perf_counter() - t0


times = []
for n_batch in n_parallel_convolution:
    ts = []
    for n_threads in n_threads_convolution:
        logger.info(f"Testing n_threads = {n_threads}, n_parallel={n_batch}")
        grid = Grid(D, l_params, N_points, n_threads=n_threads, fields_name=[], k0=k0)
        t0 = time.perf_counter()
        total_elapsed = sum(benchmark_function(grid, n_batch) for _ in range(math.ceil(number_cycles / n_batch)))
        ts.append(total_elapsed)
        logger.info(f"finished in {time.perf_counter() - t0}s")
    times.append(ts)
times = np.array(times)

logger.info(f"Benchmark times: {times}")
# Plot results
plt.subplot(1, 2, 1)
cmap = matplotlib.colormaps["jet"]
for i, n_batch in np.ndenumerate(n_parallel_convolution):
    i = i[0]
    color = cmap(i / n_parallel_convolution.size) if n_parallel_convolution.size > 1 else "black"
    if i == 0:
        plt.plot(n_threads_convolution, n_threads_convolution / np.min(n_threads_convolution), "-", label="ideal", color=color)
    plt.plot(n_threads_convolution, times[0, 0] / times[i], "o", color=color, label=f"n_batch={n_batch}")
if len(n_parallel_convolution) == 1:
    plt.axvline(x=n_threads_convolution[np.argmin(times)], linestyle="--", color="red", label="best time")
plt.ylim(top=1.1 * times[0, 0] / np.min(times))
plt.xlabel("# of threads")
plt.ylabel("Convolution speed, rescaled\n(speed=1/time)")
plt.legend()
plt.subplot(1, 2, 2)
for i, n_batch in np.ndenumerate(n_parallel_convolution):
    i = i[0]
    color = cmap(i / n_parallel_convolution.size) if n_parallel_convolution.size > 1 else "black"
    plt.semilogy(n_threads_convolution, times[i] / number_cycles, "o", color=color, label=f"n_parallel={n_batch}")
plt.xlabel("# of threads")
plt.ylabel("Avg convolution time (s)")
# noinspection PyUnboundLocalVariable
plt.suptitle(rf"Convolution speed vs parallelism (threaded and non-threaded) for N={N_points}, D={D}, k0={k0} for $\lambda={grid.l:.2f}$")
pltshowm(save="benchmarking_parallel.png", full=True)
