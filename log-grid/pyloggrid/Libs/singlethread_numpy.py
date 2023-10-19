"""
Ensures that ``numpy`` is single-threaded. Does not disrupt ``Cython`` threading.

Warning:
    Must be imported top-level, before any other library that uses ``numpy`` !!

Example:
    Replace
    ::

        import numpy as np

    by

    ::

        from pyloggrid.Libs.singlethread_numpy import np
"""
import os
import sys

from pyloggrid.Libs.custom_logger import setup_custom_logger

logger = setup_custom_logger(__name__)

if os.name == "posix":
    import resource  # Increase max recursion depth

    try:
        resource.setrlimit(resource.RLIMIT_STACK, (2**29, -1))
    except ValueError:
        logger.warning("Could not set RLIMIT_STACK")
sys.setrecursionlimit(int(1e6))

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# noinspection PyUnresolvedReferences
import numpy as np

np.array(0)  # to avoid F401 in flake8 lint
