# ETD4RK Thresholds

ETD4RK is a solver based on _Exponential time differencing for stiff systems_ (Cox & Matthews 2002), equations (26)-(29).

In its equations intervene differences of functions and their truncated Taylor series, such as $e^x-1-x$.

Due to the floating point precision of python, for small enough $x$, this difference becomes numerically unstable, and should be replaced by the higher order term of the Taylor series.

This folder contains the `thresholds.py` script which showcases both the exact numerical function and the higher-order taylor term, as a means to determine at which threshold we should switch from one to the other. Those thresholds are the ones used in `Framework.ETD4RK`.
