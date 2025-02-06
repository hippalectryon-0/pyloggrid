"""utests for singlethread_numpy.py"""

from pyloggrid.Libs.singlethread_numpy import np

np.array(0)


def test_recursion_depth():
    # noinspection PyMissingOrEmptyDocstring
    def f(N):
        if N < 0:
            return
        return f(N - 1)

    f(1e5)
