"""utests for Grid.py"""

import itertools
import os
import sys

import pytest

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from pyloggrid.Libs.singlethread_numpy import np

np.zeros(0)  # Do not remove - used to force singlethread_numpy to be imported before the other libraries when refactoring

from typing import Callable

from pyloggrid.Libs.datasci import randcomplex, randcomplex_like
from pyloggrid.Libs.misc import composed_decs
from pyloggrid.LogGrid.DataExplorer import DataExplorer
from pyloggrid.LogGrid.Framework import Solver
from pyloggrid.LogGrid.Grid import Grid

grids_cache = []


def basicGrid(N: int, D: int = 3, l_params: dict = None, k0: bool = False, cache=True, n_threads: int = 1) -> Grid:
    """a basic grid for tests"""
    if l_params is None:
        l_params = {"a": None, "b": None, "plastic": False}
    if cache:
        for grid in grids_cache:
            if grid.D == D and grid.N_points == N and grid.l_params == l_params and grid.k0 == k0:
                grid.fields_names = []
                grid.fields = {}
                print("using cache")
                return grid
    grid = Grid(D=D, l_params=l_params, N_points=N, fields_name=[], k0=k0, n_threads=n_threads)
    grids_cache.append(grid)
    return grid


def get_grid_and_array(N: int, D: int, l_params: dict = None, k0: bool = False, n_threads: int = 1) -> tuple[Grid, np.ndarray]:
    """stub for tests, returns a basic grid and a rancom array of similar shape"""
    grid = basicGrid(N=N, D=D, l_params=l_params, k0=k0, n_threads=n_threads)
    return grid, randcomplex_like(grid.ks[0])


def get_2Dgrid_nodiv(N: int, k0: bool = False, l_params: dict = None) -> tuple[Grid, np.ndarray]:
    """2D divergence-free grid with vorticity as a variable"""
    grid, ux = get_grid_and_array(N, D=2, k0=k0, l_params=l_params)
    mask = grid.ks[1] != 0
    uy = randcomplex_like(ux)
    uy[mask] = (ux * grid.ks[0])[mask] / grid.ks[1][mask]
    u = np.array([ux, uy])
    w = grid.maths.rot2D(u)
    grid.fields_names = ["w"]
    grid.init_fields()
    grid.load_fields({"w": w})
    return grid, w


def get_3Dgrid_nodiv(N: int, k0: bool = False, l_params: dict = None) -> tuple[Grid, np.ndarray]:
    """3D divergence-free grid with vorticity as a variable"""
    grid, ux = get_grid_and_array(N, D=3, k0=k0, l_params=l_params)
    uy = randcomplex_like(ux)
    mask = grid.ks[2] != 0
    uz = randcomplex_like(ux)
    uz[mask] = (ux * grid.ks[0] + uy * grid.ks[1])[mask] / grid.ks[2][mask]
    u = np.array([ux, uy, uz])
    w = grid.maths.rot3D(u)
    grid.fields_names = ["wx", "wy", "wz"]
    grid.init_fields()
    grid.load_fields({"wx": w[0], "wy": w[1], "wz": w[2]})

    return grid, w


d_D123 = pytest.mark.parametrize("D", [1, 2, 3])


def d_l_params(f):
    """decorator to test for different l_params"""

    # noinspection PyMissingOrEmptyDocstring
    def g(*args, **kwargs):
        for params in [
            {"a": None, "b": None, "plastic": False},
            {"a": 1, "b": 2, "plastic": False},
            {"a": None, "b": None, "plastic": True},
        ]:
            f(l_params=params, *args, **kwargs)

    return g


d_k0 = pytest.mark.parametrize("k0", [False, True])

d_N = pytest.mark.parametrize("N", [3, 5, 7, 10])

d_k0_N = composed_decs(d_k0, d_N)
d_D123_l_params_k0_N = composed_decs(d_D123, d_l_params, d_k0, d_N)
d_D123_k0_N = composed_decs(d_D123, d_k0, d_N)
d_D123_k0 = composed_decs(d_D123, d_k0)
d_k0_l_params = composed_decs(d_k0, d_l_params)
d_D123_l_params = composed_decs(d_D123, d_l_params)
d_D123_N = composed_decs(d_D123, d_N)
d_k0_l_params_N = composed_decs(d_k0, d_l_params, d_N)
d_l_params_N = composed_decs(d_l_params, d_N)
d_D123_k0_l_params_N = composed_decs(d_D123, d_k0, d_l_params, d_N)

args_k0 = [False, True]
args_N = [3, 5, 7, 10]
args_l_params = (
    {"a": None, "b": None, "plastic": False},
    {"a": 1, "b": 2, "plastic": False},
    {"a": None, "b": None, "plastic": True},
)
args_D = [1, 2, 3]
args_k0_N = ("k0, N", list(itertools.product(args_k0, args_N)))
args_D123_k0_N = ("D, k0, N", list(itertools.product(args_D, args_k0, args_N)))
args_D123_l_params_k0_N = ("D, k0, N, l_params", list(itertools.product(args_D, args_k0, args_N, args_l_params)))


class TestGrid:
    @pytest.mark.parametrize(*args_D123_k0_N)
    def test_init_fields(self, N, D, k0):
        n_k0 = 1 if k0 else 0
        shape = (N + n_k0,) + (D - 1) * (2 * N + n_k0,)
        fields_name = ["a", "b.乱流b", "cc cc", "teSt'"]
        fields = {k: np.zeros(shape) for k in fields_name}

        # Test before initializing fields
        grid = basicGrid(N, D=D, k0=k0)
        assert len(grid.fields) == 0

        # Test fields initialization: length, type, values
        grid.fields_names = fields_name
        grid.init_fields()
        assert len(grid.fields) == len(fields_name)
        for f in grid.fields.values():
            assert f.dtype == complex
        assert np.array_equal(grid.fields_names, fields_name)
        for k in fields_name:
            assert np.array_equal(fields[k], grid.fields[k])

    def test_get_l_from_params(self):
        grid = basicGrid(1)

        # lambda=2
        grid.l_params = {"a": None, "b": None, "plastic": False}
        l = grid.get_l_from_params()
        assert l == 2

        # lambda=Phi
        grid.l_params = {"a": 1, "b": 2, "plastic": False}
        l = grid.get_l_from_params()
        assert (l - 1.6180) / l < 0.01

        # lambda=plastic
        grid.l_params = {"a": None, "b": None, "plastic": True}
        l = grid.get_l_from_params()
        assert (l - 1.324717) / l < 0.01

    @pytest.mark.parametrize(*args_D123_k0_N)
    def test_generate_points(self, N, D, k0):
        k_min = np.random.random() * 100
        n_k0 = 1 if k0 else 0
        grid = basicGrid(N=N, D=D, k0=k0)

        # generate
        points = np.zeros((2 * N + n_k0,))
        for n in range(N):
            points[N + n + n_k0] = grid.l**n
            points[N - 1 - n] = -(grid.l**n)
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
                ks_1D_mid[i] = np.exp((np.log(v) + np.log(ks_1D[i + 1])) / 2)
        ks_1D_expected, ks_1D_mid_expected, ks_expected, ks_mod_expected = ks_1D, ks_1D_mid, ks_space, ks_modulus

        # compare
        ks_1D, ks_1D_mid, ks, ks_mod = grid.generate_points(k_min=k_min)
        if D == 1:
            assert np.array_equal(ks[0], ks_1D)
        elif D == 2:
            assert np.array_equal(ks[0][:, 0], ks_1D)
        elif D == 3:
            assert np.array_equal(ks[0][:, 0, 0], ks_1D)
        else:
            raise ValueError
        assert np.array_equal(ks, ks_expected)
        assert np.array_equal(ks_1D_expected, ks_1D)
        assert np.array_equal(ks_1D_mid_expected, ks_1D_mid)
        assert np.array_equal(ks_mod_expected, ks_mod)
        assert np.array_equal(ks_mod, np.abs(ks_mod))

    @pytest.mark.parametrize(*args_D123_k0_N)
    def test_load_fields(self, D, k0, N):
        n0 = 1 if k0 else 0
        shape = (N + n0,) + (D - 1) * (2 * N + n0,)
        fields_name = ["a", "bb", "cc cc"]
        fields = {k: randcomplex_like(np.zeros(shape)) for k in fields_name}
        # test same dimension
        grid = basicGrid(N, D=D, k0=k0)
        grid.fields_names = fields_name
        grid.init_fields()
        grid.load_fields(fields)
        assert len(grid.fields) == len(fields_name)
        for k in fields_name:
            assert np.array_equal(fields[k], grid.fields[k])

        # test loading grid wityh different size
        N_g = 5
        shape = (N_g + n0,) + (D - 1) * (2 * N_g + n0,)
        fields = {k: randcomplex_like(np.zeros(shape)) for k in fields_name}
        grid.load_fields(fields)
        if N >= N_g:
            a, b = N, N_g
            fields_a, fields_b = fields, grid.fields
        else:
            a, b = N_g, N
            fields_a, fields_b = grid.fields, fields
        for k in fields_name:
            if D == 1:
                assert np.array_equal(fields_a[k], fields_b[k][: b + n0])
            if D == 2:
                assert np.array_equal(fields_a[k], fields_b[k][: b + n0, a - b : a + b + n0])
            if D == 3:
                assert np.array_equal(fields_a[k], fields_b[k][: b + n0, a - b : a + b + n0, a - b : a + b + n0])


class TestMaths:
    """test Grid.Maths"""

    @pytest.mark.parametrize(*args_D123_l_params_k0_N)
    def test_generate_convolution_kernel(self, D, N, l_params, k0):
        grid = basicGrid(N=N, D=D, l_params=l_params, k0=k0)
        kernel = grid.maths.generate_convolution_kernel()
        assert kernel.dtype == np.uint32

    @pytest.mark.parametrize(*args_D123_k0_N)
    def test_dx(self, D, k0, N):
        grid, arr = get_grid_and_array(N, D, k0=k0)
        assert np.array_equal(grid.maths.dx * arr, 1j * grid.ks[0] * arr)

    @pytest.mark.parametrize(*args_D123_k0_N)
    def test_dy(self, D, k0, N):
        if D > 1:
            grid, arr = get_grid_and_array(N, D, k0=k0)
            assert np.array_equal(grid.maths.dy * arr, 1j * grid.ks[1] * arr)

    @pytest.mark.parametrize(*args_k0_N)
    def test_dz(self, k0, N):
        for D in [3]:
            grid, arr = get_grid_and_array(N, D, k0=k0)
            assert np.array_equal(grid.maths.dz * arr, 1j * grid.ks[2] * arr)

    @pytest.mark.parametrize(*args_D123_k0_N)
    def test_d2x(self, D, k0, N):
        grid, arr = get_grid_and_array(N, D, k0=k0)
        assert np.isclose(grid.maths.d2x * arr, -(grid.ks[0] ** 2) * arr).all()

    @pytest.mark.parametrize(*args_D123_k0_N)
    def test_d2y(self, D, k0, N):
        if D > 1:
            grid, arr = get_grid_and_array(N, D, k0=k0)
            assert np.isclose(grid.maths.d2y * arr, -(grid.ks[1] ** 2) * arr).all()

    @pytest.mark.parametrize(*args_k0_N)
    def test_d2z(self, k0, N):
        for D in [3]:
            grid, arr = get_grid_and_array(N, D, k0=k0)
            assert np.isclose(grid.maths.d2z * arr, -(grid.ks[2] ** 2) * arr).all()

    @pytest.mark.parametrize(*args_k0_N)
    def test_rot2D_inv(self, k0, N):
        grid, arr = get_grid_and_array(N, D=2, k0=k0)
        arr2 = -(arr * grid.ks[0])  # enforce div=0
        arr = arr * grid.ks[1]
        rot = grid.maths.rot2D([arr, arr2])
        assert np.isclose(grid.maths.rot2D_inv * rot, np.array([arr, arr2])).all()

    @pytest.mark.parametrize(*args_k0_N)
    def test_rot3D_inv(self, k0, N):
        grid, arr = get_grid_and_array(N, D=3, k0=k0)
        arr2 = randcomplex_like(arr)
        arr3 = -(arr * grid.ks[0] + arr2 * grid.ks[1])  # enforce div=0
        arr2 = arr2 * grid.ks[2]
        arr = arr * grid.ks[2]
        # arr[grid.k_nonzero == False], arr2[grid.k_nonzero == False], arr3[grid.k_nonzero == False] = 0, 0, 0
        rot = grid.maths.rot3D([arr, arr2, arr3])
        assert np.isclose(grid.maths.rot3D_inv(rot), [arr, arr2, arr3]).all()

    @pytest.mark.parametrize(*args_k0_N)
    def test_cross2d(self, k0, N):
        grid, arr = get_grid_and_array(N, D=2, k0=k0)
        arr1 = randcomplex((2,) + arr.shape)
        arr2 = randcomplex((2,) + arr.shape)
        assert np.array_equal(grid.maths.cross2D(arr1, arr2), arr1[0] * arr2[1] - arr1[1] * arr2[0])

    @pytest.mark.parametrize(*args_k0_N)
    def test_cross3d(self, k0, N):
        grid, arr = get_grid_and_array(N, D=3, k0=k0)
        ax, ay, az = randcomplex((3,) + arr.shape)
        bx, by, bz = randcomplex((3,) + arr.shape)
        assert np.array_equal(grid.maths.cross3D([ax, ay, az], [bx, by, bz]), [ay * bz - az * by, bx * az - bz * ax, ax * by - ay * bx])

    @pytest.mark.parametrize(*args_k0_N)
    def test_rot2d(self, k0, N):
        grid, arr = get_grid_and_array(N, D=2, k0=k0)
        arr1 = randcomplex((2,) + arr.shape)
        assert np.array_equal(grid.maths.rot2D(arr1), 1j * grid.maths.cross2D(grid.ks, arr1))

    @pytest.mark.parametrize(*args_k0_N)
    def test_rot3d(self, k0, N):
        grid, arr = get_grid_and_array(N, D=3, k0=k0)
        arr1 = randcomplex((3,) + arr.shape)
        assert np.array_equal(grid.maths.rot3D(arr1), 1j * grid.maths.cross3D(grid.ks, arr1))
        assert np.isclose(grid.maths.div3D(grid.maths.rot3D(arr1)), np.zeros_like(arr)).all()

    @pytest.mark.parametrize(*args_k0_N)
    def test_laplacian(self, k0, N):
        grid, arr = get_grid_and_array(N, D=3, k0=k0)
        kx, ky, kz = grid.ks
        assert np.isclose(grid.maths.laplacian * arr, -(kx**2 + ky**2 + kz**2) * arr).all()
        grid, arr = get_grid_and_array(N, D=2, k0=k0)
        kx, ky = grid.ks
        assert np.isclose(grid.maths.laplacian * arr, -(kx**2 + ky**2) * arr).all()
        grid, arr = get_grid_and_array(N, D=1, k0=k0)
        (kx,) = grid.ks
        assert np.isclose(grid.maths.laplacian * arr, -(kx**2) * arr).all()

    @pytest.mark.parametrize(*args_D123_k0_N)
    def test_laplacian_inv(self, k0, N, D):
        grid, arr = get_grid_and_array(N, D=D, k0=k0)
        arr[~grid.k_nonzero] = 0
        assert np.isclose(grid.maths.laplacian_inv * grid.maths.laplacian * arr, arr).all()
        assert np.isclose(grid.maths.laplacian * grid.maths.laplacian_inv * arr, arr).all()

    @pytest.mark.parametrize(*args_k0_N)
    def test_div2d(self, k0, N):
        grid, arr = get_grid_and_array(N, D=2, k0=k0)
        arr = randcomplex((2,) + arr.shape)
        assert np.isclose(grid.maths.div2D(arr), grid.maths.dx * arr[0] + grid.maths.dy * arr[1]).all()

    @pytest.mark.parametrize(*args_k0_N)
    def test_div3d(self, k0, N):
        grid, arr = get_grid_and_array(N, D=3, k0=k0)
        arr = randcomplex((3,) + arr.shape)
        assert np.isclose(grid.maths.div3D(arr), grid.maths.dx * arr[0] + grid.maths.dy * arr[1] + grid.maths.dz * arr[2]).all()

    @pytest.mark.parametrize(*args_k0_N)
    def test_inner_product(self, k0, N):
        grid, arr = get_grid_and_array(N, D=3, k0=k0)
        Mask = np.ones_like(arr)
        if k0:
            Mask[grid.ks[0] == 0] = 1 / 2
        arr2 = randcomplex_like(arr)
        assert np.isclose(grid.maths.inner_product(arr, arr2), 2 * np.real(np.sum(arr * np.conj(arr2 * Mask)))).all()
        assert grid.maths.inner_product(arr, arr2).imag == 0

    @pytest.mark.parametrize(*args_k0_N)
    def test_self_inner_product(self, k0, N):
        grid, arr = get_grid_and_array(N, D=3, k0=k0)
        assert np.array_equal(grid.maths.inner_product(arr, arr), grid.maths.self_inner_product(arr))

    @pytest.mark.parametrize(*args_D123_l_params_k0_N)
    def test_convolve(self, D, l_params, k0, N):
        # /!\ 2**18*2pi > default tolerance of np.isclose => We can't take N>=18
        if D == 2 and N > 7:
            pytest.skip("too slow")  # Too slow otherwise
        if D == 3 and N > 3:
            pytest.skip("too slow")
        grid, arr1 = get_grid_and_array(N, D=D, l_params=l_params, k0=k0)
        arr1 = grid.maths.enforce_grid_symmetry_arr(arr1)
        arr2 = grid.maths.enforce_grid_symmetry_arr(randcomplex_like(arr1))

        convolve_expected = np.zeros_like(arr1)
        ks_ordered = np.moveaxis(grid.ks, 0, -1)
        i_k_test = [
            tuple(i)
            for i in np.random.randint(
                0,
                grid.N_points,
                (
                    10,
                    D,
                ),
            )
        ]
        if D == 1:
            i_k_test += [
                (0,),
                (grid.N_points - 1,),
            ]
        elif D == 2:
            i_k_test += [
                (0, grid.N_points),
                (1, grid.N_points),
                (grid.N_points // 2, grid.N_points),
                (0, 1),
                (0, grid.N_points // 2),
            ]
        elif D == 3:
            i_k_test += [
                (0, grid.N_points, grid.N_points),
                (1, grid.N_points, grid.N_points),
                (grid.N_points // 2, grid.N_points, grid.N_points),
                (0, 1, 1),
                (0, grid.N_points // 2, 1),
            ]
        i_k_test = tuple(set(i_k_test))  # select unique
        for i_k in i_k_test:
            k = ks_ordered[i_k]
            print(f"Testing convolution on k={k}")
            found = set()
            for i_p in np.ndindex(ks_ordered.shape[:-1]):
                for i_q in np.ndindex(ks_ordered.shape[:-1]):
                    p, q = ks_ordered[i_p], ks_ordered[i_q]
                    if np.isclose(p + q, k).all():
                        id_ = (*p, *q)
                        if id_ in found:
                            continue
                        found.add(id_)
                        convolve_expected[i_k] = convolve_expected[i_k] + arr1[i_p] * arr2[i_q]
                    if np.isclose(p - q, k).all():
                        id_ = (*p, *[-i for i in q])
                        if id_ in found:
                            continue
                        found.add(id_)
                        convolve_expected[i_k] = convolve_expected[i_k] + arr1[i_p] * arr2[i_q].conjugate()
                    if np.isclose(-p + q, k).all():
                        id_ = (*[-i for i in p], *q)
                        if id_ in found:
                            continue
                        found.add(id_)
                        convolve_expected[i_k] = convolve_expected[i_k] + arr1[i_p].conjugate() * arr2[i_q]

        expected = [convolve_expected[c] for c in i_k_test]
        got = [grid.maths.convolve(arr1, arr2)[c] for c in i_k_test]
        assert np.isclose(got, expected).all()

    @pytest.mark.parametrize(*args_D123_l_params_k0_N)
    def test_convolve_batch(self, D, l_params, k0, N):
        for _ in range(1, 4):
            grid, arr1 = get_grid_and_array(N, D=D, l_params=l_params, k0=k0)

            for N_batch in range(1, 10):
                fgs = [
                    (
                        grid.maths.enforce_grid_symmetry_arr(randcomplex_like(arr1)),
                        grid.maths.enforce_grid_symmetry_arr(randcomplex_like(arr1)),
                    )
                    for _ in range(N_batch)
                ]
                expected = np.array([grid.maths.convolve(f, g) for f, g in fgs])
                results = grid.maths.convolve_batch(fgs)

                assert np.isclose(results, expected).all()

    @pytest.mark.parametrize(*args_D123_l_params_k0_N)
    def test_convolve_properties(self, D, l_params, k0, N):
        grid, arr1 = get_grid_and_array(N, D=D, l_params=l_params, k0=k0)
        convolve = grid.maths.convolve
        arr1 = grid.maths.enforce_grid_symmetry_arr(arr1)
        arr2 = grid.maths.enforce_grid_symmetry_arr(randcomplex_like(arr1))

        assert np.isclose(convolve(arr1, arr2), convolve(arr2, arr1)).all()  # commutativity

        arr3 = grid.maths.enforce_grid_symmetry_arr(randcomplex_like(arr1))
        assert np.isclose(
            grid.maths.inner_product(arr1, convolve(arr2, arr3)), grid.maths.inner_product(convolve(arr1, arr2), arr3)
        )  # associativity in average
        assert np.isclose(grid.maths.inner_product(arr1, convolve(arr2, arr3)), grid.maths.inner_product(convolve(arr1, arr3), arr2))


class TestPhysics:
    @pytest.mark.parametrize(*args_k0_N)
    def test_enstrophy(self, N, k0):
        grid, w = get_3Dgrid_nodiv(N=N, k0=k0)

        assert np.isclose(grid.physics.enstrophy(), grid.maths.inner_product(w, w) / 2).all()
        assert grid.physics.enstrophy().imag == 0

    @pytest.mark.parametrize(*args_D123_k0_N)
    def test_energy(self, D, N, k0):
        match D:
            case 3:
                grid, w = get_3Dgrid_nodiv(N=N, k0=k0)
                u = grid.maths.rot3D_inv(w)
            case 2:
                grid, w = get_2Dgrid_nodiv(N=N, k0=k0)
                u = grid.maths.rot2D_inv * w
            case 1:
                grid, u = get_grid_and_array(N, D=1, k0=k0)
                grid.fields_names = ["u"]
                grid.init_fields()
                grid.load_fields({"u": u})
            case _:
                raise ValueError

        assert np.isclose(grid.physics.energy(), grid.maths.self_inner_product(u) / 2).all()
        assert grid.physics.energy().imag == 0

    @pytest.mark.parametrize(*args_k0_N)
    def test_helicity(self, N, k0):
        grid, w = get_3Dgrid_nodiv(N=N, k0=k0)
        u = grid.maths.rot3D_inv(w)

        assert np.isclose(grid.physics.helicity(), grid.maths.inner_product(u, w)).all()
        assert grid.physics.helicity().imag == 0

    @pytest.mark.parametrize(*args_D123_l_params_k0_N)
    def test_spectrum(self, k0, D, l_params, N):
        if D == 1:
            return
        n_k0 = 1 if k0 else 0
        if D == 2:
            grid, _ = get_2Dgrid_nodiv(N=N, k0=k0, l_params=l_params)

            # noinspection PyMissingOrEmptyDocstring
            def cust_quantity(fields: dict, pts: np.ndarray):
                w = fields["w"]
                ux, _ = grid.maths.rot2D_inv * w
                return (ux[pts] * np.imag(w[pts])).real

        elif D == 3:
            grid, _ = get_3Dgrid_nodiv(N=N, k0=k0, l_params=l_params)

            # noinspection PyMissingOrEmptyDocstring
            def cust_quantity(fields: dict, pts: np.ndarray):
                wx, wy, wz = fields["wx"], fields["wy"], fields["wz"]
                ux, _, __ = grid.maths.rot3D_inv([wx, wy, wz])
                return (ux[pts] * np.imag(wy[pts])).real

        else:
            raise ValueError(D)
        spectrum = grid.physics.spectrum(cust_quantity)
        spectrum_expected = np.zeros((grid.N_points + n_k0,))
        ks = grid.ks_1D
        for i, k in enumerate(ks):
            pts = (grid.ks_modulus == 0) if k == 0 else (k <= grid.ks_modulus) & (grid.ks_modulus < grid.l * k * 0.99)  # 0.99 to avoid overflow
            delta = (grid.l - 1) * k * np.sum(pts) if k != 0 else 1
            res = np.sum(cust_quantity(grid.fields, pts))
            assert res.imag == 0, "The callback provided in the spectrum returned imaginary results"
            if delta > 0:
                spectrum_expected[i] += res / delta
        assert np.isclose(spectrum, spectrum_expected).all()


class Test3DEuler:
    """
    Simulate the 3D Euler equation. Checks that the energy is conserved and the divergence is zero.
    """

    def test_3D_euler(self, tmp_path):
        E0 = [None]

        # noinspection PyMissingOrEmptyDocstring
        def equation_nl(_t: float, grid: Grid, _simu_params: dict) -> dict[str, np.ndarray]:
            M = grid.maths

            ux, uy, uz = grid.field("ux", "uy", "uz")

            # Evolution
            uxdxux = M.convolve(ux, M.dx * ux)
            uydyux = M.convolve(uy, M.dy * ux)
            uxdxuy = M.convolve(ux, M.dx * uy)
            uydyuy = M.convolve(uy, M.dy * uy)
            uzdzux = M.convolve(uz, M.dz * ux)
            uzdzuy = M.convolve(uz, M.dz * uy)
            uxdxuz = M.convolve(ux, M.dx * uz)
            uydyuz = M.convolve(uy, M.dy * uz)
            uzdzuz = M.convolve(uz, M.dz * uz)

            # Evolution w/o pressure
            dux_dt = -uxdxux - uydyux - uzdzux
            duy_dt = -uxdxuy - uydyuy - uzdzuy
            duz_dt = -uxdxuz - uydyuz - uzdzuz

            # Add pressure
            dux_dt, duy_dt, duz_dt = grid.maths.P_projector([dux_dt, duy_dt, duz_dt])

            return {"ux": dux_dt, "uy": duy_dt, "uz": duz_dt}

        # noinspection PyMissingOrEmptyDocstring
        def equation_l(_t: float, grid: Grid, _simu_params: dict) -> dict[str, np.ndarray]:
            ux, uy, uz = grid.field("ux", "uy", "uz")

            # Checks
            assert np.linalg.norm(grid.maths.div3D([ux, uy, uz])) < 1e-10, "The divergence should be vanishingly small"
            E = grid.physics.energy()
            if E0[0] is None:
                E0[0] = E
            assert np.abs(E - E0) / E0 < 1e-3, "The energy should be conserved"

            return {"ux": np.zeros_like(ux), "uy": np.zeros_like(ux), "uz": np.zeros_like(ux)}

        # noinspection PyMissingOrEmptyDocstring
        def initial_conditions(fields: dict[str, np.ndarray], grid: Grid, _simu_params: dict) -> dict[str, np.ndarray]:
            grid = grid.to_new_size_empty(N_points)
            ks = grid.ks_modulus

            np.random.seed(1337)
            w = randcomplex_like(grid.ks)
            w[:, ks >= grid.k_min * grid.l**3] = 0
            ux, uy, uz = grid.maths.rot3D_inv(w)

            assert np.max(np.abs(grid.maths.div3D([ux, uy, uz]))) < 1e-10, "The divergence should be vanishingly small"

            fields["ux"] = ux
            fields["uy"] = uy
            fields["uz"] = uz

            return fields

        fields = ["ux", "uy", "uz"]
        D = 3
        l_params = {"plastic": False, "a": None, "b": None}
        n_threads_convolution = 1
        N_points = 10

        # Do not edit below unless you know what you're doing #

        simu_params = {}
        save_path = f"{tmp_path}/results/save_3D"

        solver = Solver(
            fields_names=fields,
            equation_nl=equation_nl,
            equation_l=equation_l,
            D=D,
            l_params=l_params,
            simu_params=simu_params,
            n_threads=n_threads_convolution,
        )
        solver.solve(save_path=save_path, initial_conditions=initial_conditions, end_simulation={"ode_step": 100})

    def test_3D_NS(self, tmp_path):
        # noinspection PyMissingOrEmptyDocstring
        def equation_nl(_: float, grid: Grid, _simu_params: dict) -> dict[str, np.ndarray]:
            M = grid.maths

            ux, uy, uz = grid.field("ux", "uy", "uz")

            # Evolution
            uxdxux = M.convolve(ux, M.dx * ux)
            uydyux = M.convolve(uy, M.dy * ux)
            uxdxuy = M.convolve(ux, M.dx * uy)
            uydyuy = M.convolve(uy, M.dy * uy)
            uzdzux = M.convolve(uz, M.dz * ux)
            uzdzuy = M.convolve(uz, M.dz * uy)
            uxdxuz = M.convolve(ux, M.dx * uz)
            uydyuz = M.convolve(uy, M.dy * uz)
            uzdzuz = M.convolve(uz, M.dz * uz)

            # Forcing
            fwx = np.ones_like(ux) / grid.ks_modulus**5
            fwy = np.ones_like(ux) / grid.ks_modulus**5
            fwz = np.ones_like(ux) / grid.ks_modulus**5
            f0 = 1e7
            fx, fy, fz = grid.maths.rot3D_inv([fwx, fwy, fwz]) * f0

            # Evolution w/o pressure
            dux_dt = -uxdxux - uydyux - uzdzux + fx
            duy_dt = -uxdxuy - uydyuy - uzdzuy + fy
            duz_dt = -uxdxuz - uydyuz - uzdzuz + fz

            # Add pressure
            dux_dt, duy_dt, duz_dt = grid.maths.P_projector([dux_dt, duy_dt, duz_dt])

            return {"ux": dux_dt, "uy": duy_dt, "uz": duz_dt}

        # noinspection PyMissingOrEmptyDocstring
        def equation_l(_t: float, grid: Grid, simu_params: dict) -> dict[str, np.ndarray]:
            Re = simu_params["Re"]

            visc = grid.maths.laplacian / Re

            return {"ux": visc, "uy": visc, "uz": visc}

        # noinspection PyMissingOrEmptyDocstring
        def initial_conditions(fields: dict[str, np.ndarray], grid: Grid, _simu_params: dict) -> dict[str, np.ndarray]:
            grid = grid.to_new_size_empty(N_points)
            ks = grid.ks_modulus

            np.random.seed(1337)
            w = randcomplex_like(grid.ks)
            w[:, ks >= grid.k_min * grid.l**3] = 0
            ux, uy, uz = grid.maths.rot3D_inv(w)

            fields["ux"] = ux
            fields["uy"] = uy
            fields["uz"] = uz

            return fields

        fields = ["ux", "uy", "uz"]
        D = 3
        l_params = {"plastic": False, "a": None, "b": None}
        n_threads_convolution = 1
        N_points = 10

        Re = 1e3

        # Do not edit below unless you know what you're doing #

        simu_params = {"Re": Re}
        save_path = f"{tmp_path}/results/save_3D"

        solver = Solver(
            fields_names=fields,
            equation_nl=equation_nl,
            equation_l=equation_l,
            D=D,
            l_params=l_params,
            simu_params=simu_params,
            n_threads=n_threads_convolution,
        )
        solver.solve(save_path=save_path, initial_conditions=initial_conditions, end_simulation={"ode_step": 100})

        def get_epsilons(grid: Grid, t: float, simu_params: dict) -> dict:
            r"""eps_inj, eps_diss, E
            /!\ f and visc hardcoded !!"""
            M = grid.maths
            Re = simu_params["Re"]
            ux, uy, uz = grid.field("ux", "uy", "uz")
            u = [ux, uy, uz]

            # /!\ hardcoded
            fwx = np.ones_like(ux) / grid.ks_modulus**5
            fwy = np.ones_like(ux) / grid.ks_modulus**5
            fwz = np.ones_like(ux) / grid.ks_modulus**5
            f0 = 1e7
            f = grid.maths.rot3D_inv([fwx, fwy, fwz]) * f0
            visc = grid.maths.laplacian / Re

            eps_inj = M.inner_product(f, u)
            eps_diss = M.inner_product(u, visc * u)

            return {"t": t, "E": grid.physics.energy(), "eps_inj": eps_inj, "eps_diss": eps_diss}

        def plot_epsilons(drawables: Callable) -> None:
            """eps_inj, eps_diss vs dtE"""
            t, eps_inj, eps_diss, E = drawables("t", "eps_inj", "eps_diss", "E")
            dtE = np.gradient(E, edge_order=2) / np.gradient(t, edge_order=2)
            # from matplotlib import pyplot as plt
            # plt.plot(t, eps_inj + eps_diss)
            # plt.plot(t, dtE)
            # plt.show()

            avg_error = np.mean(np.abs((dtE - (eps_inj + eps_diss)) / dtE))
            assert avg_error < 0.05, "dtE=eps_inj+eps_diss isn't verified within 5%"

        draw_funcs = {
            "epsilons": {"get": get_epsilons, "plot": plot_epsilons},
        }

        dexp = DataExplorer(save_path)
        dexp.display(draw_funcs=draw_funcs, N_points=1000, n_jobs=1)
