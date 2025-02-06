"""utests for datasci.py"""

from numpy.testing import assert_allclose

from pyloggrid.Libs.singlethread_numpy import np

np.zeros(0)  # Do not remove - used to force singlethread_numpy to be imported before the other libraries when refactoring
from typing import Optional

from pyloggrid.Libs.datasci import (
    clamp_zero,
    filter_range,
    fit,
    logmean,
    mean,
    powerlaw_fit,
    ragged_array_to_array,
    rand_seeded_by_array,
    randcomplex,
    randcomplex_like,
    randcomplex_seeded_by_array,
)


def test_filter_range():
    N = 100
    test_array = np.random.normal(size=(N,))

    # test start
    t1 = filter_range(test_array, start=0.1)
    assert np.array_equal(t1, test_array[int(N * 0.1) :])

    # test end
    t2 = filter_range(test_array, end=0.1)
    assert np.array_equal(t2, test_array[: int(N * 0.1)])

    # test both
    t3 = filter_range(test_array, start=0.3, end=0.2)
    assert np.array_equal(t3, test_array[int(N * 0.3) : int(N * 0.2)])

    # test none
    t4 = filter_range(test_array)
    assert np.array_equal(t4, test_array)

    # test overlapping
    t5 = filter_range(test_array, start=0.8, end=0.8)
    assert np.array_equal(t5, [])


def test_mean():
    N = 100
    test_data = np.random.normal(size=(N,))
    test_ts = np.abs(np.random.normal(size=(N,)))

    # useful
    weights = np.gradient(test_ts)
    test_data[test_data == 0] = 0.1

    # test mean with data
    assert mean(test_data)[0] == np.mean(test_data)
    assert mean(test_data)[1] == np.std(test_data)

    # test log mean
    assert mean(test_data, log=True)[0] == np.exp(np.mean(np.log(np.abs(test_data))))

    # test weighted mean
    assert mean(test_data, test_ts)[0] == np.average(test_data, weights=weights)

    # test log weighted mean
    assert mean(test_data, test_ts, log=True)[0] == np.exp(np.average(np.log(np.abs(test_data)), weights=weights))

    # test zeros
    data_zeros = np.zeros_like(test_data)
    assert mean(data_zeros)[0] == 0

    # test weights with zeros
    ts_zeros = np.ones_like(test_data)
    ts_zeros[0] = 0
    data_zeros[-1] = 1
    assert mean(data_zeros, ts_zeros)[0] == 0


def test_clamp_zero():
    N = 100
    test_data = np.random.normal(size=(N,))

    # test clamping with no zero values
    data = np.ones_like(test_data)
    assert np.array_equal(clamp_zero(data), data)

    # test clamping with all zeros
    data = np.zeros_like(test_data)
    assert np.array_equal(clamp_zero(data), np.ones_like(data) * np.nextafter(0, 1))

    # test clamping with near-zero values
    data = np.zeros_like(test_data)
    data[0] = np.nextafter(0, 1) * 3
    expected = np.array(data)
    expected[expected == 0] = np.nextafter(0, 1)
    assert np.array_equal(clamp_zero(data), expected)

    # test clamping with big values
    data = np.zeros_like(test_data)
    data[:3] = [2, 1, 3]
    expected = np.array(data)
    expected[expected == 0] = 1 / 10
    assert np.array_equal(clamp_zero(data), expected)

    # test clamping with both zeros, near-min values and nonzero values
    data = np.zeros_like(test_data)
    data[:6] = [2, 1, 3, 0, 0, np.nextafter(0, 1) * 3]
    expected = np.array(data)
    expected[expected == 0] = np.nextafter(0, 1)
    assert np.array_equal(clamp_zero(data), expected)

    # test clamping with empty array
    data = np.array([])
    expected = np.array(data)
    assert np.array_equal(clamp_zero(data), expected)


class TestFit:
    def test_linear_fit(self):
        # Generate linear data with some noise
        x = np.linspace(0, 1, num=500)
        y = 2 * x + 1 + np.random.normal(scale=0.1, size=x.shape)

        # Define the linear function
        def linear(x, a, b):
            return a * x + b

        # Call the fit function
        params = fit(linear, x, y)

        # Check that the fitted parameters are close to the expected values
        assert_allclose(params, [2, 1], rtol=0.1)

    def test_quadratic_fit(self):
        # Generate quadratic data with some noise
        x = np.linspace(-5, 5, num=500)
        y = 3 * x**2 + 2 * x + 1 + np.random.normal(scale=0.1, size=x.shape)

        # Define the quadratic function
        def quadratic(x, a, b, c):
            return a * x**2 + b * x + c

        params1 = fit(quadratic, x, y, mask=(0.1, 0.9))
        assert_allclose(params1, [3, 2, 1], rtol=0.1)

        params2 = fit(quadratic, x, y, mask=x > 0)
        assert_allclose(params2, [3, 2, 1], rtol=0.1)


def test_powerlaw_fit():
    N = 100
    x = np.linspace(1, 5, N)

    def test_powerlaw(a_real: float, b_real: float, y: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None) -> None:
        """test a powerlaw y = b * x ** a"""
        if y is None:
            y = b_real * x**a_real
        b, a = powerlaw_fit(x, y * (1 + np.random.normal(size=x.shape) / 100), mask=mask)
        assert np.abs(a - a_real) / a_real < 0.1
        assert np.abs(b - b_real) / b_real < 0.1

    # test simple power law
    test_powerlaw(2.3, 4.1)

    # negative exponent
    test_powerlaw(1.8, 3.3)

    # test mask
    y = 9.1 * x**2.5
    y[x.size // 2 :] = 1
    test_powerlaw(2.5, 9.1, y=y, mask=np.array([i < x.size // 2 for i in range(x.size)]))

    # test fixed a
    a_real = 3.82
    b_real = 9.1
    y = b_real * x**a_real
    b, _ = powerlaw_fit(x, y * (1 + np.random.normal(size=x.shape) / 100), a=a_real)
    assert np.abs(b - b_real) / b_real < 0.1


def test_randcomplex_like():
    shape = (23, 56, 12)
    data = randcomplex_like(np.zeros(shape))

    assert data.shape == shape
    assert data.dtype == complex


def test_randcomplexe():
    shape = (23, 56, 12)
    data = randcomplex(shape)

    assert data.shape == shape
    assert data.dtype == complex


def test_randcomplex_seeded_by_array():
    res = randcomplex_seeded_by_array(np.zeros((15, 15)), 111)
    assert res.dtype == complex


def test_rand_seeded_by_array():
    source = np.eye(15)
    res = rand_seeded_by_array(source, 111)
    assert res.dtype == float
    for val in np.unique(source):
        assert np.unique(res[source == val]).size == 1

    # Test very big sources
    source = (np.random.random((15, 15)) - 0.5) * 1e250
    res = rand_seeded_by_array(source, 111)
    for val in np.unique(source):
        assert np.unique(res[source == val]).size == 1

    # Test changing size
    source1 = np.random.randint(0, 10, (7, 7))
    res1 = rand_seeded_by_array(source1, 111)
    source2 = np.random.randint(5, 11, (14, 14))
    res2 = rand_seeded_by_array(source2, 111)
    for val in np.unique(source1):
        u = np.unique(res2[source2 == val])
        if u.size == 0:
            continue
        assert u.size == 1
        assert u[0] == res1[source1 == val][0]


class TestRaggedArrayToArray:
    def test_ragged_array_to_array(self):
        # Test the function with valid input
        input_array = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        expected_output = np.array([[1.0, 2.0, 3.0, np.nan], [4.0, 5.0, np.nan, np.nan], [6.0, 7.0, 8.0, 9.0]])
        result = ragged_array_to_array(input_array)
        assert_allclose(result, expected_output, equal_nan=True)

    def test_ragged_array_to_array_empty_list(self):
        # Test the function with an empty list as input
        input_array = []
        expected_output = np.array([])
        result = ragged_array_to_array(input_array)
        assert_allclose(result, expected_output)

    def test_ragged_array_to_array_already_complete_array(self):
        # Test the function with an already complete 2D array as input
        input_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected_output = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = ragged_array_to_array(input_array)
        assert_allclose(result, expected_output)


class TestLogMean:
    def test_logmean(self):
        # Test the function with valid input
        input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(float)
        expected_output = np.array([1.817121, 4.932424, 7.958114])
        result = logmean(input_data, axis=1)
        assert_allclose(result, expected_output, rtol=1e-4, atol=1e-4)

    def test_logmean_with_zero_values(self):
        # Test the function with input containing zero values
        input_data = np.array([[1, 2, 3], [0, 5, 6], [7, 8, 0]]).astype(float)
        expected_output = np.array([1.817121, 5.477226, 7.483315])
        result = logmean(input_data, axis=1)
        assert_allclose(result, expected_output, rtol=1e-4, atol=1e-4)

    def test_logmean_with_all_nan_values(self):
        # Test the function with input containing all NaN values
        input_data = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])
        expected_output = np.array([np.nan, np.nan, np.nan])
        result = logmean(input_data, axis=1)
        assert_allclose(result, expected_output, equal_nan=True)
