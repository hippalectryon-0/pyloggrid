"""
run a benchmark for current build

Ran as ``run_benchmark.py flags N_cycles N``
"""
import sys
import time
from pathlib import Path

import numpy as np
import numpy.ctypeslib as npct
from matplotlib import pyplot as plt

# noinspection PyProtectedMember
from pyloggrid.LogGrid.Grid import _setup_convolver_c

convolver_c = npct.load_library("convolver_c", ".")

_setup_convolver_c(convolver_c)

N_cycles = float(sys.argv[2])
N = int(sys.argv[3])


def do_benchmark(N: int = 13, N_cycles: float = 1e4, show: bool = True) -> None:
    """do a benchmark
    Args:
        N: the grid size
        N_cycles: the number of convolutions to do
        show: if True, plot a hist of results
    """
    from pyloggrid.LogGrid.Grid import Grid

    def do_cycle(params: dict) -> None:
        """the part to benchmark"""
        grid, f, g = params["grid"], params["f"], params["g"]
        convolver_c.convolve(grid.maths.convolution_kernel.flatten(), grid.maths.convolution_kernel.size, f, g, np.zeros_like(f))

    def init_cycle(params: dict) -> dict:
        """called before starting a cycle"""
        grid = params["grid"]
        params["f"] = np.random.randn(*grid.ks_modulus.shape).astype("complex").flatten()
        params["g"] = np.random.randn(*grid.ks_modulus.shape).astype("complex").flatten()
        return params

    # This is where you set which parameter is benchmarked
    def init_newparam() -> dict:
        """called after changing the param's value"""
        l_params = {"plastic": False, "a": 1, "b": 2}
        n_threads = 1
        D = 3
        k0 = False

        grid = Grid(D, l_params, N, n_threads=n_threads, fields_name=[], k0=k0)

        return {"grid": grid}

    cycles_times = []
    params = init_newparam()
    for _ in range(int(N_cycles)):
        params = init_cycle(params)
        start = time.perf_counter()
        do_cycle(params)
        cycles_times.append(time.perf_counter() - start)

    if show:
        plt.hist(cycles_times, bins=int(np.sqrt(N_cycles)), density=True)
        plt.axvline(x=np.mean(cycles_times), color="black")
        plt.show()

    # save
    flags = sys.argv[1]
    if flags.startswith("[]"):  # default, can be run several timesq
        flags = flags.replace("[]", f"[DEFAULT_{np.random.randint(0, 9999)}]")
    Path("results").mkdir(exist_ok=True)
    np.save(f"results/flags={flags}", cycles_times)


if __name__ == "__main__":
    do_benchmark(show=False, N_cycles=N_cycles, N=N)
