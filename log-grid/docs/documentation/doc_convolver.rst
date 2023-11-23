*************
The convolver
*************

.. important:: This is an *advanced* page. You do not need to understand this to use the framework.

.. note:: To optimize the convolver's performance, see :doc:`doc_performance`.

This page documents details of the implementation of the convolution.

General workflow
################

Let's consider two fields ``f, g`` on a :class:`~pyloggrid.LogGrid.Grid.Grid`.
The convolution of those two function is defined as :math:`\displaystyle f*g=\sum_{p+q=k}f(p)g(q)`.

To optimize the computation of this sum, before starting a simulation, we compute a *convolution kernel*, i.e. a map of all the ``(k,p,q)`` where ``p+q=k``.

Since the convolution is the most computationally expensive part of the simulations, it is optimized in C in ``pyloggrid/LogGrid/convolver_c.c``.
For similar reasons, the computation of the convolution kernel is compiled from Cython in ``pyloggrid/LogGrid/compute_convolution_kernel.pyx``.

Convolution kernel
##################

``compute_convolution_kernel.pyx`` exposes one function for each ``D=1,2,3`` dimention: ``compute_interaction_kernel_[123]D``.

The script is written in such a way that computing the kernel is done with the same code for ``k0=True`` and ``k0=False``.

Although optimizing this part is not *critical* (since it's only run once), we kept this optimized code from earlier versions of the framework.
The speedup is much appreciated for large grids.

.. note:: Computing the convolution kernel only once at the start provides a speedup during each convolution compared to computing it on-the-fly. However, it also requires keeping the whole kernel in RAM, which for big grids can take several Gbs. If you require very large computation kernels that exceed your RAM, feel free to open a pull request/issue so that we can discuss adding a way to compute the kernel on-the-fly.

Convolution
###########

Interfacing C and python
************************

The compiled C code is imported to python via :mod:`numpy.ctypeslib`.
It is also important to specify which C inputs the imported functions accept to avoid segfaults. This is done via :func:`~pyloggrid.LogGrid.Grid.Maths._setup_convolver_c`.

Default convolving functions
============================

As explained `here <C code>`_, the C file exposes a number of different functions to convolve fields.

Based on internal benchmarks, we chose different functions for different situations based on parrallelism.

Batches
-------

* To convolve two functions ``f, g``, :func:`~pyloggrid.LogGrid.Grid.Maths.convolve` uses ``convolver_c.convolve`` if ``n_threads==1``, else ``convolver_c.convolve_omp``.
* To convolve a list of function couples ``[(f1, g1), g(2, g2), ...]``, :func:`~pyloggrid.LogGrid.Grid.Maths.convolve_batch` we use:

    * if ``n_threads==1``:

        * ``convolver_c.convolve_list_batch_V2`` for 2 couples
        * ``convolver_c.convolve_list_batch_V3`` for 3 couples
        * ``convolver_c.convolve_list_batch_V4`` for 4+ couples
    * otherwise it uses :func:`~pyloggrid.LogGrid.Grid.Maths._convolve_batch_list`, which is faster than ``convolver_c.convolve_list_batch_V[234]_omp``.

.. _C code:

C code
======

Although we don't offer compile PyLogGrid for MSVC on Windows (see :ref:`Windows vs Linux`), ``convolver_c.c`` is written to be compilable with both Clang, GCC and MINGW.

Functions containing ``_V`` are designed to take advantage of `AVX <https://en.wikipedia.org/wiki/Advanced_Vector_Extensions>`_.

In particular, ``convolve_list_batch_V[N=1,2,3,4]_omp`` compute ``N`` convolutions in parallel. Although in theory some architectures could benefit from ``N=8`` and higher, we found no practical benefit on our end.






