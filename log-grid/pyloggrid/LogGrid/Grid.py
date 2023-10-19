"""Class that directly handles the log grid & related maths"""
import typing

if typing.TYPE_CHECKING:  # for Sphinx
    import numpy as np

    np.zeros(0)

from pyloggrid.Libs.singlethread_numpy import np

np.zeros(0)  # Do not remove - used to force singlethread_numpy to be imported before the other libraries when refactoring

import ctypes as ct
import logging
import math as libmath
import os
import pathlib
from typing import Callable, Iterable, Union

import numpy.ctypeslib as npct
import scipy.optimize

# noinspection PyProtectedMember
from numpy._typing import ArrayLike

import pyloggrid.LogGrid.compute_convolution_kernel as conv_kernel_generator
from pyloggrid.Libs.custom_logger import setup_custom_logger
from pyloggrid.Libs.misc import TimeTracker, bytes2human

logger = setup_custom_logger(__name__, level=logging.INFO)

convolver_c = npct.load_library("convolver_c", pathlib.Path(os.path.abspath(__file__)).parent)


def _setup_convolver_c() -> None:
    """Setup the imported C function. Important to avoid segfaults."""

    npflags = ["C_CONTIGUOUS"]  # Require a C contiguous array in memory
    # noinspection PyTypeChecker
    ui32_1d_type = npct.ndpointer(dtype=np.uint32, ndim=1, flags=npflags)
    # noinspection PyTypeChecker
    complex_1d_type = npct.ndpointer(dtype=complex, ndim=1, flags=npflags)

    args = [ui32_1d_type, ct.c_uint32, complex_1d_type, complex_1d_type, complex_1d_type]
    convolver_c.convolve.argtypes = args
    convolver_c.convolve_omp.argtypes = args

    args = [ui32_1d_type, ct.c_uint32, ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_void_p), ct.c_uint32, ct.c_short, complex_1d_type]
    convolver_c.convolve_list_omp.argtypes = args
    convolver_c.convolve_list_batch_V2.argtypes = args
    convolver_c.convolve_list_batch_V3.argtypes = args
    convolver_c.convolve_list_batch_V4.argtypes = args
    convolver_c.convolve_list_batch_V2_omp.argtypes = args
    convolver_c.convolve_list_batch_V3_omp.argtypes = args
    convolver_c.convolve_list_batch_V4_omp.argtypes = args

    convolver_c.set_omp_threads.argtypes = [ct.c_uint32]


_setup_convolver_c()


class Grid:
    """Main class that handles the log grid"""

    def init_fields(self) -> None:
        """Creates the arrays corresponding to the field names"""
        dict_fields = {field: np.zeros(self.shape, dtype="complex") for field in self.fields_names}
        self.fields = dict_fields

    def get_l_from_params(self) -> float:
        """Sets the grid spacing from the grid's parameters.

        Returns:
            the grid spacing

        Reference:
            Campolina & Mailybaev 2020, *Fluid dynamics on logarithmic lattices*
        """
        if self.l_params["plastic"]:  # lambda=plastic
            return 1.324717957244746025
        a, b = self.l_params["a"], self.l_params["b"]
        if a is None and b is None:  # lambda=2
            return 2
        # lambda^b-lambda^a=1
        assert b > a, "grid parameter `b` should be greater than `a`"
        assert a > 0, "grid parameter `a` should be >0"
        assert libmath.gcd(a, b) == 1, "grid parameters `a, b` should be irreductible (gcd=1)"
        assert (a != 1 or b != 3) and (a != 4 or b != 5), "(a,b) = (1,3) or (4,5) correspond to plastic number solutions"
        # noinspection PyUnresolvedReferences
        return float(scipy.optimize.fsolve(lambda x: x**b - x**a - 1, np.array(2))[0])

    def __init__(
        self,
        D: int,
        l_params: dict[str, Union[float, bool]],
        N_points: int,
        fields_name: list[str],
        k_min: float = None,
        k0: bool = False,
        n_threads: int = None,
    ):
        """
        Args:
            D: space dimension
            l_params: params that define the grid spacing ``{a, b, plastic: bool}``. For ``l=2``, chose ``a=b=None``. ``plastic`` supercedes all other
            N_points: length of the grid among the smallest axis
            k_min: minimum k on the grid
            fields_name: name of the fields. Used for ordering
            k0: whether the mode ``k=0`` exists
            n_threads: number of threads used in convolutions
        """
        # default val
        if k_min is None:
            k_min = 2 * np.pi
        if n_threads is None:
            try:
                n_threads = int(len(os.sched_getaffinity(0)) * 0.5)
            except AttributeError:
                n_threads = 5

        self.D = D
        self.l_params = l_params
        self.l = self.get_l_from_params()
        logger.debug(f"Chose l={self.l} from params {self.l_params}")
        self.N_points = N_points
        self.k_min = k_min
        self.k0 = k0
        self.ks_1D, self.ks_1D_mid, self.ks, self.ks_modulus = self.generate_points(k_min)
        self.shell_pts = np.zeros((self.ks_1D.size, 2), dtype=object)
        for i, k in enumerate(self.ks_1D):
            pts = (self.ks_modulus == 0) if k == 0 else (k <= self.ks_modulus) & (self.ks_modulus < self.l * k * 0.99)  # 0.99 to avoid overflow
            self.shell_pts[i] = [pts, np.sum(pts)]
        self.k_nonzero = self.ks_modulus > 0
        self.k_alongk0axis = self.ks[0] == 0
        self.maths = self.Maths(self, n_threads)
        self.physics = self.Physics(self)
        self.time_tracker: TimeTracker = TimeTracker()

        self.shape = self.ks[0].shape
        self.fields_names = fields_name
        self.fields = None
        self.init_fields()  # init fields with zeros

        self.L = 2 * np.pi / self.k_min
        self.V = self.L**self.D

    def generate_points(self, k_min: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create the grid's wavevector arrays

        Args:
            k_min: the minimum wavevector

        Returns:
            (array of wave vectors along X axis ``[N,]``, logmean of array of wave vectors along X axis [N,], array of wave vectors ``[D, N, (2N, 2N)]``, array of wave vector modulus ``[N, (2N, 2N)]``)
        """
        D = self.D
        N = self.N_points
        l = self.l

        n_k0 = 1 if self.k0 else 0

        points = np.zeros((2 * N + n_k0,))
        for n in range(N):
            points[N + n + n_k0] = l**n
            points[N - 1 - n] = -(l**n)
        points = np.array(points) * k_min

        ks_space = [np.zeros((2 * N + n_k0,) * D)]
        for p in range(len(points)):
            ks_space[0][p] = points[p]
        ks_space.extend(np.swapaxes(ks_space[0], 0, i) for i in range(1, D))
        ks_space = np.array([ks[N:] for ks in ks_space])
        ks_modulus = np.sqrt(np.sum([ks_i**2 for ks_i in ks_space], axis=0))

        ks_1D = points[N:]
        ks_1D_mid = np.zeros_like(ks_1D)
        for i, v in enumerate(ks_1D):
            if i + 1 == ks_1D.size:
                ks_1D_mid[i] = np.exp((np.log(v) + np.log(np.max(ks_modulus))) / 2)
            else:
                ks_1D_mid[i] = 0 if v == 0 else np.exp((np.log(v) + np.log(ks_1D[i + 1])) / 2)

        return ks_1D, ks_1D_mid, ks_space, ks_modulus

    def load_fields(self, fields: dict[str, np.ndarray]) -> "Grid":
        """Load fields from a grid of another dimension. If the new grid is smaller, discard outer fields. If bigger, set new fields to 0.

        Args:
            fields: fields from the previous grid. fields names must correspond for both grids, as well as ``k_min``.
        """
        for field_name, field in self.fields.items():
            N2 = fields[field_name].shape[0]
            n_k0 = 1 if self.k0 else 0
            N = self.N_points + n_k0
            if self.D == 1:
                self.fields[field_name][: min(N2, N)] = fields[field_name][: min(N2, N)]
            elif self.D == 2:
                self.fields[field_name][: min(N2, N), N - min(N2, N) : N + min(N2, N) - n_k0] = fields[field_name][
                    : min(N2, N), N2 - min(N2, N) : N2 + min(N2, N) - n_k0
                ]
            elif self.D == 3:
                self.fields[field_name][: min(N2, N), N - min(N2, N) : N + min(N2, N) - n_k0, N - min(N2, N) : N + min(N2, N) - n_k0] = fields[field_name][
                    : min(N2, N), N2 - min(N2, N) : N2 + min(N2, N) - n_k0, N2 - min(N2, N) : N2 + min(N2, N) - n_k0
                ]
        return self

    def enforce_grid_symmetry(self) -> None:
        """Force the ``f(-k)=f(k).conj`` symmetries along 0 axes for all fields"""
        self.maths.enforce_grid_symmetry_dict(self.fields)

    class Maths:
        """Functions to perform maths on log grids"""

        cached_conv_kernel = None

        def __init__(self, grid: "Grid", n_threads: int = 0):
            self.grid = grid
            self.n_threads = n_threads
            convolver_c.set_omp_threads(
                n_threads
            )  # In theory we want to minimize the number of calls to this (slow), but we're not supposed to regenerate this object each step so we should be ok
            self.convolve_c = convolver_c.convolve if self.n_threads == 1 else convolver_c.convolve_omp
            self.convolve_c_batch_V = (
                {
                    2: convolver_c.convolve_list_batch_V2,
                    3: convolver_c.convolve_list_batch_V3,
                    4: convolver_c.convolve_list_batch_V4,
                }
                if self.n_threads == 1
                else {
                    2: convolver_c.convolve_list_batch_V2_omp,
                    3: convolver_c.convolve_list_batch_V3_omp,
                    4: convolver_c.convolve_list_batch_V4_omp,
                }
            )
            self.convolve_batch = self.convolve_batch_V if self.n_threads == 1 else self.convolve_batch_list  # list is faster when parallelized (for now)

            self.convolution_kernel = self.generate_convolution_kernel()  # Kernel for convolutions

        def generate_convolution_kernel(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            """Generate the convolution kernel (who interacts with who) for log grid convolutions

            Returns:
                (kernel offsets, kernel signs) | signs are used to determine when to take the complex conjugate (+1 = normal, -1 = conjugate)
            """
            if (
                self.cached_conv_kernel is not None
                and self.cached_conv_kernel["N"] == self.grid.N_points
                and self.cached_conv_kernel["l_params"] == self.grid.l_params
                and self.cached_conv_kernel["k0"] == self.grid.k0
                and self.cached_conv_kernel["D"] == self.grid.D
            ):
                return self.cached_conv_kernel["kernel"]
            a, b = self.grid.l_params["a"], self.grid.l_params["b"]
            if not self.grid.l_params["plastic"] and (a is None and b is None):  # l = 2
                kernel_offsets = [(1, 0), (0, 1), (-1, -1)]
                kernel_signs = [(1, -1), (-1, 1), (1, 1)]
            elif not self.grid.l_params["plastic"]:  # a, b
                kernel_offsets = [(b, a), (a, b), (b - a, -a), (-a, b - a), (-b, a - b), (a - b, -b)]

                kernel_signs = [(1, -1), (-1, 1), (1, -1), (-1, 1), (1, 1), (1, 1)]

            else:  # plastic
                kernel_offsets = [(3, 1), (1, 3), (2, -1), (-1, 2), (-3, -2), (-2, -3), (5, 4), (4, 5), (1, -4), (-4, 1), (-5, -1), (-1, -5)]

                kernel_signs = [(1, -1), (-1, 1), (1, -1), (-1, 1), (1, 1), (1, 1), (1, -1), (-1, 1), (1, -1), (-1, 1), (1, 1), (1, 1)]

            # k=0
            i_list = list(range(-self.grid.ks[0].shape[0] + 1, self.grid.ks[0].shape[0]))
            i_list.remove(0)
            kernel_offsets_k0_ax0 = np.array([(0, 0)] + [(i, i) for i in i_list] * 2)
            kernel_signs_k0_ax0 = np.array(
                [(1, 1)]
                + [(1, -1) for _ in range((kernel_offsets_k0_ax0.shape[0] - 1) // 2)]
                + [(-1, 1) for _ in range((kernel_offsets_k0_ax0.shape[0] - 1) // 2)]
            )
            if self.grid.D > 1:
                i_list = list(range(-self.grid.ks[0].shape[1] + 1, self.grid.ks[0].shape[1]))
                i_list.remove(0)
                kernel_offsets_k0_ax1 = np.array([(0, 0)] + [(i, i) for i in i_list] * 2)
                kernel_signs_k0_ax1 = np.array(
                    [(1, 1)]
                    + [(1, -1) for _ in range((kernel_offsets_k0_ax1.shape[0] - 1) // 2)]
                    + [(-1, 1) for _ in range((kernel_offsets_k0_ax1.shape[0] - 1) // 2)]
                )
            else:
                kernel_offsets_k0_ax1, kernel_signs_k0_ax1 = 1, 1
            if self.grid.k0:
                kernel_offsets += [(0, 0), (0, 0)]
                kernel_signs += [(1, 0), (0, 1)]
            # convert for Cython
            kernel_offsets, kernel_signs, kernel_offsets_k0_ax0, kernel_signs_k0_ax0, kernel_offsets_k0_ax1, kernel_signs_k0_ax1 = (
                np.array(kernel_offsets, dtype=np.short),
                np.array(kernel_signs, dtype=np.short),
                np.array(kernel_offsets_k0_ax0, dtype=np.short),
                np.array(kernel_signs_k0_ax0, dtype=np.short),
                np.array(kernel_offsets_k0_ax1, dtype=np.short),
                np.array(kernel_signs_k0_ax1, dtype=np.short),
            )

            if self.grid.D == 1:
                # noinspection PyTypeChecker
                kernel = conv_kernel_generator.compute_interaction_kernel_1D(
                    kernel_offsets, kernel_signs, kernel_offsets_k0_ax0, kernel_signs_k0_ax0, self.grid.k0, self.grid.ks[0].shape[0]
                )
            elif self.grid.D == 2:
                # noinspection PyTypeChecker
                kernel = conv_kernel_generator.compute_interaction_kernel_2D(
                    kernel_offsets,
                    kernel_signs,
                    kernel_offsets_k0_ax0,
                    kernel_signs_k0_ax0,
                    kernel_offsets_k0_ax1,
                    kernel_signs_k0_ax1,
                    self.grid.k0,
                    self.grid.ks[0].shape[0],
                    self.grid.ks[0].shape[1],
                )
            elif self.grid.D == 3:
                # noinspection PyTypeChecker
                kernel = conv_kernel_generator.compute_interaction_kernel_3D(
                    kernel_offsets,
                    kernel_signs,
                    kernel_offsets_k0_ax0,
                    kernel_signs_k0_ax0,
                    kernel_offsets_k0_ax1,
                    kernel_signs_k0_ax1,
                    self.grid.k0,
                    self.grid.ks[0].shape[0],
                    self.grid.ks[0].shape[1],
                    self.grid.ks[0].shape[2],
                )
            else:
                raise ValueError
            # noinspection PyUnresolvedReferences
            logger.info(f"Convolution coeffs generated - approx {kernel.shape[0]} triads, kernel size approx {bytes2human(kernel.size * kernel.itemsize)}")
            Grid.Maths.cached_conv_kernel = {"N": self.grid.N_points, "l_params": self.grid.l_params, "D": self.grid.D, "k0": self.grid.k0, "kernel": kernel}
            # noinspection PyTypeChecker
            return kernel

        # actual operations below
        @property
        def dx(self) -> np.ndarray:
            """x-derivative"""
            return 1j * self.grid.ks[0]

        @property
        def dy(self) -> np.ndarray:
            """y-derivative"""
            return 1j * self.grid.ks[1]

        @property
        def dz(self) -> np.ndarray:
            """z-derivative"""
            return 1j * self.grid.ks[2]

        @property
        def d2x(self) -> np.ndarray:
            """2nd x-derivative"""
            return self.dx * self.dx

        @property
        def d2y(self) -> np.ndarray:
            """2nd y-derivative"""
            return self.dy * self.dy

        @property
        def d2z(self) -> np.ndarray:
            """2nd z-derivative"""
            return self.dz * self.dz

        def convolve(self, f: np.ndarray, g: np.ndarray) -> np.ndarray:
            """Convolve the two arrays.

            Returns:
                the convolved array
            """
            with self.grid.time_tracker("convolution"):
                f_flat = f.flatten()
                res = np.zeros_like(f_flat)
                # noinspection PyUnresolvedReferences
                with self.grid.time_tracker("convolution_C"):
                    # noinspection PyUnresolvedReferences
                    self.convolve_c(self.convolution_kernel, self.convolution_kernel.size, f_flat, g.flatten(), res)
                res = res.reshape(f.shape)

            return res

        def convolve_batch_V_inner(self, fgs: list[tuple[np.ndarray, np.ndarray]], V: int) -> np.ndarray:
            """Inner function for <convolve_batch_V>. Convolves in batch V couples, calling the corresponding ``convolve_list_batch_V`` function in C."""
            if V == 1:
                f, g = fgs[0][0], fgs[0][1]
                f_flat = f.flatten()
                res = np.zeros_like(f_flat)
                with self.grid.time_tracker("convolution_C"):
                    # noinspection PyUnresolvedReferences
                    self.convolve_c(self.convolution_kernel, self.convolution_kernel.size, f_flat, g.flatten(), res)
                return res
            fsize = fgs[0][0].size
            N_batch = len(fgs)
            res = np.zeros((fsize * N_batch,), dtype=fgs[0][0].dtype)

            f_flats, g_flats = [], []
            for f, g in fgs:
                f_flats.append(f.flatten())
                g_flats.append(g.flatten())

            arr_f, arr_g = (ct.c_void_p * N_batch)(), (ct.c_void_p * N_batch)()  # create array of pointers
            for i in range(N_batch):
                arr_f[i], arr_g[i] = f_flats[i].ctypes.data_as(ct.c_void_p), g_flats[i].ctypes.data_as(ct.c_void_p)

            with self.grid.time_tracker("convolution_C"):
                # noinspection PyUnresolvedReferences
                self.convolve_c_batch_V[V](self.convolution_kernel, self.convolution_kernel.size, arr_f, arr_g, fsize, N_batch, res)

            return res

        def convolve_batch_V(self, fgs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
            """from couples ``((f0, g0), f(f1, g1), ...)``, computes their convolution

            Returns:
                the convolved arrays ``f0*g0``, ``f1*g1``, etc.
            """
            with self.grid.time_tracker("convolution"):
                max_V = 4  # relevant for most processors. This is not threading ! This is how many convolutions we compute "at the same time" on one thread.
                N_batch = len(fgs)
                N_V, diff_V = N_batch // max_V, N_batch % max_V
                res_maxV, res_diffV = np.array([], dtype=fgs[0][0].dtype), np.array([], dtype=fgs[0][0].dtype)
                if N_V > 0:
                    res_maxV = self.convolve_batch_V_inner(fgs[: N_V * max_V], max_V)
                if diff_V > 0:
                    res_diffV = self.convolve_batch_V_inner(fgs[-diff_V:], diff_V)

                res = np.concatenate((res_maxV, res_diffV)).reshape((len(fgs),) + fgs[0][0].shape)
            return res

        def convolve_batch_list_slower(self, fgs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
            """from couples ``((f0, g0), f(f1, g1), ...)``, computes their convolution

            Returns:
                the convolved arrays ``f0*g0``, ``f1*g1``, etc.
            """
            with self.grid.time_tracker("convolution"):
                fsize = fgs[0][0].size
                N_batch = len(fgs)
                res = np.zeros((fsize * N_batch,), dtype=fgs[0][0].dtype)

                f_flats, g_flats = [], []
                for f, g in fgs:
                    f_flats.append(f.flatten())
                    g_flats.append(g.flatten())

                arr_f, arr_g = (ct.c_void_p * N_batch)(), (ct.c_void_p * N_batch)()  # create array of pointers
                for i in range(N_batch):
                    arr_f[i], arr_g[i] = f_flats[i].ctypes.data_as(ct.c_void_p), g_flats[i].ctypes.data_as(ct.c_void_p)

                with self.grid.time_tracker("convolution_C"):
                    # noinspection PyUnresolvedReferences
                    convolver_c.convolve_list_omp(self.convolution_kernel, self.convolution_kernel.size, arr_f, arr_g, fsize, N_batch, res)

                res = res.reshape((len(fgs),) + fgs[0][0].shape)
            return res

        def convolve_batch_list(self, fgs: list[tuple[np.ndarray, np.ndarray]]) -> Iterable[np.ndarray]:
            """from couples ``((f0, g0), f(f1, g1), ...)``, computes their convolution

            Returns:
                the convolved arrays ``f0*g0``, ``f1*g1``, etc.
            """
            return [self.convolve(f, g) for (f, g) in fgs]

        @staticmethod
        def cross2D(a: ArrayLike, b: ArrayLike) -> np.ndarray:
            """2D cross product"""
            ax, ay = np.array(a)
            bx, by = b
            return ax * by - ay * bx

        @staticmethod
        def cross3D(a: ArrayLike, b: ArrayLike) -> np.ndarray:
            """3D cross product"""
            ax, ay, az = np.array(a)
            bx, by, bz = b
            return np.array([ay * bz - az * by, bx * az - bz * ax, ax * by - ay * bx])

        def rot2D(self, a: ArrayLike) -> np.ndarray:
            """2D rotational"""
            return 1j * self.cross2D(self.grid.ks, a)

        def rot3D(self, a: ArrayLike) -> np.ndarray:
            """3D rotational"""
            return 1j * self.cross3D(self.grid.ks, a)

        @property
        def rot2D_inv(self) -> np.ndarray:
            """Inverse rotational for 2D fields, *assuming div = 0*"""
            return np.array([-self.dy * self.laplacian_inv, self.dx * self.laplacian_inv])

        def rot3D_inv(self, a: ArrayLike) -> np.ndarray:
            """Inverse rotational for 3D fields, *assuming div = 0*"""
            return -self.laplacian_inv * self.rot3D(a)

        @property
        def laplacian(self) -> np.ndarray:
            """laplacian"""
            return -self.grid.ks_modulus**2

        @staticmethod
        def inv(f: ArrayLike) -> np.ndarray:
            """Inverse of a linear operator.

            it's assumed that where the operator is zero, the function to invert will also be zero
            """
            f = np.array(f)
            nz = f != 0
            f[nz] = 1 / f[nz]
            return f

        @property
        def laplacian_inv(self) -> np.ndarray:
            """inverse laplacian"""
            return self.inv(self.laplacian)

        def div3D(self, a: ArrayLike) -> np.ndarray:
            """3D divergence"""
            ax, ay, az = a
            return self.dx * ax + self.dy * ay + self.dz * az

        def div2D(self, a: ArrayLike) -> np.ndarray:
            """2D divergence"""
            ax, ay = a
            return self.dx * ax + self.dy * ay

        def inner_product(self, f: ArrayLike, g: ArrayLike) -> float:
            """Log grid inner product"""
            mask = np.ones_like(g)
            f = np.array(f)
            if self.grid.k0:
                if len(f.shape) == len(self.grid.ks[0].shape) + 1:
                    mask[:, self.grid.k_alongk0axis] = 1 / 2
                    # TODO pourquoi 2*1/2 et pas 1* sur la moitié des points ? Et on divise pas à tort le point (0,0) par 2 par hasard ?
                else:
                    mask[self.grid.k_alongk0axis] = 1 / 2
            return 2 * np.sum(f * np.conj(g * mask)).real

        def self_inner_product(self, f: ArrayLike) -> float:
            """inner product applied on oneself"""
            return self.inner_product(f, f)  # noqa

        def P_projector(self, A: ArrayLike) -> np.ndarray:
            """Turns ``dUdt = A - gradP`` into ``dUdt = A'``

            Args:
                A: ``dUdt`` without the pressure term

            Returns:
                ``A'``
            """
            ks, ks_modulus = self.grid.ks, self.grid.ks_modulus
            Calc = np.zeros_like(A, dtype="complex")
            if len(Calc.shape) == len(self.grid.ks[0].shape) + 1:
                Calc[:, self.grid.k_nonzero] = (ks * np.sum(A * ks, axis=0))[:, self.grid.k_nonzero] / ks_modulus[self.grid.k_nonzero] ** 2
            else:
                Calc[self.grid.k_nonzero] = (ks * np.sum(A * ks, axis=0))[self.grid.k_nonzero] / ks_modulus[self.grid.k_nonzero] ** 2

            return A - Calc

        def enforce_grid_symmetry_arr(self, arr: ArrayLike) -> np.ndarray:
            """Force the ``f(-k)=f(k).conj`` symmetries along ``k=0`` axes. Modifies the array in-place"""
            arr = np.array(arr)
            grid = self.grid
            D = grid.D
            arr[grid.ks_modulus == 0] = arr[grid.ks_modulus == 0].real  # enfore symmetry
            if D > 1:
                arr[(grid.ks[0] == 0) & (grid.ks[1] < 0)] = arr[(grid.ks[0] == 0) & (grid.ks[1] > 0)][::-1].conjugate()
            if D > 2:
                arr[(grid.ks[0] == 0) & (grid.ks[1] == 0) & (grid.ks[2] < 0)] = arr[(grid.ks[0] == 0) & (grid.ks[1] == 0) & (grid.ks[2] > 0)][::-1].conjugate()

            return arr

        def enforce_grid_symmetry_dict(self, fields: dict) -> dict:
            """forces symmetry for all fields of dict, in-place"""
            if self.grid.k0:  # only enforce if we have k0 mode to avoid slowing down
                for field_name, field in fields.items():
                    fields[field_name] = self.enforce_grid_symmetry_arr(field)

            return fields

    class Physics:
        """Computes a few classical physical properties"""

        def __init__(self, grid: "Grid"):
            self.grid = grid

        def enstrophy(self) -> float:
            """enstrophy Ω=ω²/2"""
            if self.grid.D != 3:
                raise NotImplementedError("1D/2D isn't coded yet")

            if "wx" in self.grid.fields:
                wx = self.grid.fields["wx"]
                wy = self.grid.fields["wy"]
                wz = self.grid.fields["wz"]
                w = [wx, wy, wz]
            else:
                ux = self.grid.fields["ux"]
                uy = self.grid.fields["uy"]
                uz = self.grid.fields["uz"]
                w = self.grid.maths.rot3D([ux, uy, uz])
            Omega = self.grid.maths.self_inner_product(w) / 2
            return Omega.real

        def energy(self) -> float:
            """kinetic energy E=u²/2"""
            match self.grid.D:
                case 3:
                    if "wx" in self.grid.fields:
                        wx = self.grid.fields["wx"]
                        wy = self.grid.fields["wy"]
                        wz = self.grid.fields["wz"]
                        w = [wx, wy, wz]

                        u = self.grid.maths.rot3D_inv(w)
                    else:
                        ux = self.grid.fields["ux"]
                        uy = self.grid.fields["uy"]
                        uz = self.grid.fields["uz"]
                        u = [ux, uy, uz]
                case 2:
                    if "w" in self.grid.fields:
                        w = self.grid.fields["w"]
                        u = self.grid.maths.rot2D_inv * w
                    else:
                        ux = self.grid.fields["ux"]
                        uy = self.grid.fields["uy"]
                        u = [ux, uy]
                case 1:
                    u = self.grid.fields["u"]
                case _:
                    raise ValueError

            E = self.grid.maths.self_inner_product(u) / 2
            return E.real

        def helicity(self) -> complex:
            """helicity H=u*ω"""
            if self.grid.D != 3:
                raise NotImplementedError("1D/2D isn't coded yet")

            wx = self.grid.fields["wx"]
            wy = self.grid.fields["wy"]
            wz = self.grid.fields["wz"]
            w = np.array([wx, wy, wz])

            u = self.grid.maths.rot3D_inv(w)
            H = self.grid.maths.inner_product(u, w)
            return H.real

        def spectrum(self, fun: Callable[[dict[str, np.ndarray], np.ndarray], np.ndarray]) -> np.ndarray:
            """Spectrum for the quantity calculated by the callback
            Callback takes two args: ``fields`` and ``ks``. The first one is ``grid.fields``. The second one is a grid-shaped bool array corresponding to concerned ``ks``.
            The callback should return a summable np.ndarray ``[observable(k) for k in ks]``.
            Ex kinetic energy 2D: ``fun(fields, ks) = ux[k]*conj(ux[k]) + uy[k]*conj(uy[k])``

            Returns:
                np.ndarray of the spectrum along each point of
                grid.ks_1D
            TODO update utest
            """
            return self.compute_by_shell(fun, normalize=True)

        def cumulative_k(self, fun: Callable[[dict[str, np.ndarray], np.ndarray], np.ndarray]) -> np.ndarray:
            """Compute the cumulative quantity between wave number 0 and k, defined by the sum over |k'| < k of that quantity
            TODO add utest
            """
            return np.cumsum(self.compute_by_shell(fun))

        def compute_by_shell(self, fun: Callable[[dict[str, np.ndarray], np.ndarray], np.ndarray], normalize: bool = False) -> np.ndarray:
            """Structure function for the quantity calculated by the callback
            Callback takes two args: ``fields`` and ``ks``. The first one is ``grid.fields``. The second one is a grid-shaped bool array corresponding to concerned ``ks``.
            The callback should return a summable np.ndarray ``[observable(k) for k in ks]``.
            Ex kinetic energy 2D: ``fun(fields, ks) = ux[k]*conj(ux[k]) + uy[k]*conj(uy[k])``

            Args:
                fun: the function to compute
                normalize: if True, normalize the result on each shell

            Returns:
                np.ndarray of the structure function along each point of
                grid.ks_1D
            TODO: add utest
            """
            grid = self.grid
            result_by_shell = np.zeros((grid.N_points + (1 if grid.k0 else 0),))

            for i, k in enumerate(grid.ks_1D):
                pts, N_k = grid.shell_pts[i]
                if N_k == 0:
                    continue
                res = np.sum(fun(grid.fields, pts))
                if normalize:
                    res /= (grid.l - 1) * k * N_k if k != 0 else 1
                assert res.imag == 0, "compute_by_shell: got imaginary results"
                result_by_shell[i] += res

            return result_by_shell

    def field(self, *args) -> list[np.ndarray] | np.ndarray:
        """get fields by name"""
        return [self.fields[i] for i in args] if len(args) > 1 else self.fields[args[0]]

    def to_new_size(self, fields: dict) -> "Grid":
        """return a new grid with the same parameters as this one but a new size

        Args:
            fields: the new fields
        """
        N_points = next(iter(fields.values())).shape[0] - (1 if self.k0 else 0)
        grid = self.to_new_size_empty(N_points)
        grid.fields = fields
        grid.time_tracker = self.time_tracker
        return grid

    def to_new_size_empty(self, N_points: int) -> "Grid":
        """return a new grid with the same parameters as this one but a new size, empty

        Args:
            N_points: the new size
        """
        grid = Grid(
            D=self.D, N_points=N_points, fields_name=self.fields_names, l_params=self.l_params, n_threads=self.maths.n_threads, k_min=self.k_min, k0=self.k0
        )
        grid.time_tracker = self.time_tracker
        return grid
