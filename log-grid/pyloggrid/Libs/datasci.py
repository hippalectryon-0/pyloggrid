"""Libraries for data science"""
import typing
from typing import Any, Callable, Iterable, Optional

if typing.TYPE_CHECKING:  # for sphinx
    import numpy as np

    np.zeros(0)

from pyloggrid.Libs.singlethread_numpy import np

np.zeros(0)  # Do not remove - used to force singlethread_numpy to be imported before the other libraries when refactoring
import scipy.optimize


def filter_range(data: np.ndarray, start: float = 0, end: float = 1) -> np.ndarray:
    """Trims a 1D array from its first and last portions.

    This is a shortcut to ``data[int(size * start) : int(size * end)]``

    Args:
        data: the data to filter
        start: where to start in the values (ex: 0.2 -> we ommit the 20% first values)
        end: where to end in the values (ex: 0.8 -> we ommit the 20% last values)

    Returns:
        the new array *(if filtered, the shape is altered)*
    """
    N = data.size
    return data[int(N * start) : int(N * end)]


def mean(data: np.ndarray, ts: Optional[np.ndarray] = None, log: bool = False, start: float = 0, end: float = 1) -> tuple[np.ndarray, np.ndarray | None]:
    """Mean of ``data``. If ``ts`` is provided, the mean is weighted by the time intervals. If ``log``, the log mean is returned.

    Args:
        data: data to average
        ts: times associated with the data. Must be increasing.
        log: if True, only the mean avg is returned ``exp(mean(log(abs(data_weighted)))``, and the variance returns ``None``
        start: where to start in the values (ex: 0.2 -> we ommit the 20% first values)
        end: where to end in the values (ex: 0.2 -> we ommit the 20% last values)

    Returns:
        the mean and std value
    """

    weights = np.ones_like(data) if ts is None else np.gradient(ts)
    weights = filter_range(weights, start=start, end=end)
    data = filter_range(data, start=start, end=end)

    if log:
        return np.exp(np.average(np.log(np.abs(data)), weights=weights)), None

    avg = np.average(data, weights=weights)
    variance = np.sqrt(np.average((data - avg) ** 2, weights=weights))  # https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    return avg, variance


def clamp_zero(data: np.ndarray, fill: Any = None) -> np.ndarray:
    """Replace all zero values with ``fill`` (defaults to ``min(data)/10``)

    Args:
        data: array to clamp
        fill: if specified, all zero values are filled with this

    Returns:
        The clamped array
    """
    data = np.array(data)
    if data.size == 0:
        return data
    if np.max(np.abs(data)) == 0:
        fill = np.nextafter(0, 1)

    data[data == 0] = max(np.min(np.abs(data[data != 0])) / 10, np.nextafter(0, 1)) if fill is None else fill

    return data


def fit(f: Callable, x: np.ndarray, y: np.ndarray, mask: np.ndarray | tuple[float, float] = None) -> tuple[float, ...]:
    """Fit ``(x,y)`` by the function ``f`` with an optional mask

    Args:
        f: fit function
        x
        y
        mask: an array mask, or a tuple (start, end)
    """
    if isinstance(mask, tuple):
        x = filter_range(x, mask[0], mask[1])
        y = filter_range(y, mask[0], mask[1])
    elif mask is not None:
        x = x[mask]
        y = y[mask]
    mask2 = np.isfinite(y)
    x = x[mask2]
    y = y[mask2]
    assert len(x) > 1, "Your fitted data must be of size >1, check your mask and NaN data"

    # noinspection PyUnresolvedReferences
    params, _ = scipy.optimize.curve_fit(f, x, y)
    return params


def powerlaw_fit(x: np.ndarray, y: np.ndarray, mask: np.ndarray | float = None, a: float = None) -> tuple[float, float]:
    """Power law fit ``Y = b * X ^ a``

    Args:
        x: X array
        y: Y array
        mask: mask to specify the relevant values of X, Y to use (same as :func:`fit`)
        a: optional fixed value for ``a``. If specified, only ``b`` is fitted

    Returns:
        the fit parameters ``b, a``
    """
    if a is None:
        lnb, a = fit(lambda logx, lnb, a: lnb + a * logx, np.log(x), np.log(y), mask)
    else:
        lnb = fit(lambda logx, lnb: lnb + a * logx, np.log(x), np.log(y), mask)
    return np.exp(lnb), a


def randcomplex_like(data: np.ndarray) -> np.ndarray:
    """Random array of complex numbers, see :func:`randcomplex`

    Args:
        data: array to copy the shape from

    Returns:
        random array
    """
    return randcomplex(data.shape)


def randcomplex(shape: Iterable) -> np.ndarray:
    """Random array of complex numbers, normally distributed in modulus and uniformly in angle

    Args:
        shape: return array shape

    Returns:
        random array
    """
    # noinspection PyTypeChecker
    return np.random.normal(size=shape) * np.exp(1.0j * np.random.uniform(0, 2 * np.pi, shape))


def rand_seeded_by_array(source: np.ndarray, seed: int) -> np.ndarray:
    r"""Returns a pseudorandom array of same shape as ``source`` where items with the same ``source`` will have the same value for a given seed.

    This is useful when creating pseudorandom arrays based on ``kx,ky,kz``, of different sizes, e.g. when changing the grid size but keeping the same pseudorandom forcing

    Warning:
        This is not optimised for speed, and should probably not be called every step
    """
    res = np.zeros_like(source)
    values, indexes = np.unique(source, return_index=True)

    for v in values:
        np.random.seed(int(seed + v) % (2**30))
        res[source == v] = np.random.random()

    return res


def randcomplex_seeded_by_array(source: np.ndarray, seed: int) -> np.ndarray:
    """Like :func:`rand_seeded_by_array`, but returns a complex array (uniform in a square)"""
    return rand_seeded_by_array(source, seed) + 1j * rand_seeded_by_array(source, seed + 1)


def ragged_array_to_array(data: np.ndarray | list, fill=np.nan) -> np.ndarray:
    """Transform an 1D array of 1D array of different sizes into an array of array of same sizes, i.e a classic 2D array.

    Smaller arrays are padded with ``fill``.

    If a classic 2D array is given, it does nothing.

    Args:
        data: array of array of different sizes
        fill: fill value

    Returns:
        padded 2D-array
    """
    if len(data) == 0:
        return np.array(data)
    len_max = len(max(data, key=len))
    for ite in range(len(data)):
        fill_array = np.empty(len_max - len(data[ite]))
        fill_array[:] = fill
        data[ite] = np.append(data[ite], fill_array)
    len_x = len(data)
    len_y = len(data[0])
    data = np.concatenate(data)
    data = data.reshape((len_x, len_y))
    return data


def logmean(data: np.ndarray, axis=None) -> np.ndarray:
    """Logarithmic mean along an axis, ignoring zero and nan values."""
    return np.exp(np.nanmean(np.log(clamp_zero(data, np.NaN)), axis=axis))
