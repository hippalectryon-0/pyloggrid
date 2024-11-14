#####################################
Welcome to PyLogGrid's documentation!
#####################################

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Getting started & Help

    loggrids
    tutorial/tutorial
    documentation/documentation
    autoapi/index

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Other

    benchmark
    whats-new


About
#####

.. image:: /static/img/PyLogGrid.svg
  :alt: <PyLogGrid logo>

PyLogGrid is a Python-based framework for running and analyzing numerical simulations on :doi:`log-lattices<10.1088/1361-6544/abef73>` [1]_. The log-lattice structure is particularly useful for modeling phenomena that exhibit multi-scale behavior, such as turbulence. PyLogGrid is designed to be flexible, customizable, and easy to use.

This framework has been used in several scientific papers such as [2]_, [3]_.

The framework includes a variety of built-in tools for analyzing simulation results, including visualization tools and post-processing scripts.

A barebones matlab framework by another research team can be found `here <https://arxiv.org/abs/2006.00047>`_ [4]_.

--------

The philosophy
==============

Sparse Fourier simulations
--------------------------

.. image:: /static/img/loggrid_white.svg
  :width: 200
  :alt: <image: A log grid>

Simulate complex systems spanning multiple scales with our numerical library. Gain insights previously unattainable with classical simulations.

Designed for physicists
-----------------------

.. image:: /static/img/newton.svg
  :width: 200
  :alt: <image: Newton>

Our library provides an intuitive and user-friendly interface for simulating and analyzing complex systems, so you can focus on your research rather than coding.

Python + C for High Performance
-------------------------------

.. image:: /static/img/computer.svg
  :width: 200
  :alt: <image: A computer>


Our library combines the ease of use and readability of Python with the speed and performance of compiled C. Get the best of both worlds for optimized execution speed.


.. [1] Campolina, C. S., & Mailybaev, A. A. (2021). Fluid dynamics on logarithmic lattices. Nonlinearity, 34(7), 4684. doi:10.1088/1361-6544/abef73
.. [2] Barral, A., & Dubrulle, B. (2023). Asymptotic ultimate regime of homogeneous Rayleigh–Bénard convection on logarithmic lattices. Journal of Fluid Mechanics, 962, A2. doi:10.1017/jfm.2023.204
.. [3] Costa, G., Barral, A., & Dubrulle, B. (2023). Reversible Navier-Stokes equation on logarithmic lattices. Physical Review E, 107(6), 065106. doi:10.1103/PhysRevE.107.065106
.. [4] Campolina, C. S. (2020). LogLatt: A computational library for the calculus and flows on logarithmic lattices. arXiv:2006.00047


