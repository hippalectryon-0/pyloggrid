"""run a benchmark for current build"""
import sys
import time

import numpy as np
import numpy.ctypeslib as npct
from matplotlib import pyplot as plt

convolver_c = npct.load_library("convolver_c", ".")  # ../../Utils/
import ctypes as ct


def define_arguments():
    """Convenience function for defining the arguments of the functions inside the imported module."""

    # Define the arguments accepted by the C functions. This is not strictly necessary,
    # but it is good practice for avoiding segmentation faults.
    npflags = ["C_CONTIGUOUS"]  # Require a C contiguous array in memory
    ui32_1d_type = npct.ndpointer(dtype=np.uint32, ndim=1, flags=npflags)
    complex_1d_type = npct.ndpointer(dtype=complex, ndim=1, flags=npflags)

    args = [ui32_1d_type, ct.c_uint32, complex_1d_type, complex_1d_type, complex_1d_type]
    convolver_c.convolve.argtypes = args


define_arguments()


def do_benchmark(N=13, N_cycles=1e3, show=True):
    """benchmark"""
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
    np.save(f"results/flags={flags}", cycles_times)


do_benchmark(show=False, N_cycles=1e4, N=15)
