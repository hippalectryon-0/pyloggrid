# Utils

## Benchmarking

`benchmarking.py` is a script to determine how many threads you should use on a given computer for a given grid.
It computes, for one convolution, the speed gain from multithreading vs the theoretical maximum gain.

###Joblib

Remark on the added benefit of doing multiple convolutions at once using Joblib: somehow, this doesn't seen to be able to provide any speed-up.
Using multiprocessing is much slower, since the arrays (kernels etc.) have to be shared across CPUs (also one could try using shared memory for this, see https://stackoverflow.com/questions/14124588/shared-memory-in-multiprocessing)
Using multithreading seems to being absolutely no speedup (same run time as with one convolution at a time, even with n_threads=1).

This is not a big deal, since we ideally want to run several simulations at once rather than just once "fast" simulation, so we'd rather use
several cores for several simulations than several cores for one slightly faster simulation.

## Profiling

`profile.bat` is a Windows profiling utility to check for execution speed bottlenecks. Since it uses pprofile, it can profile individual lines.
Requirement: [qcachegrind](https://sourceforge.net/projects/qcachegrindwin/)

To profile the Cython code, some flags must be added:
* on top of the `.pyx` file, declare `#cython: linetrace=True`
* in the `setup.py`, add:

```python
ext_modules = [
    Extension(...,
              define_macros=[('CYTHON_TRACE_NOGIL', '1')]
              )
]
from Cython.Compiler.Options import get_directive_defaults
directive_defaults = get_directive_defaults()
directive_defaults['linetrace'] = True
directive_defaults['binding'] = True
setup(...)
```
If you want to see the line profiling, put a copy of the convolver's source in the running directory.
Then simply run `profile.bat` and the results should open in qcachegrind.
