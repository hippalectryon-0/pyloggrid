*********************
The different solvers
*********************

Why traditional solvers don't work
##################################

Traditional solvers, such as Runge-Kutta, don't work out-of-the-box for log grids.
This is because as the wavenumbers follow exponential progressions :math:`k\sim\lambda^n`, and thus the viscosity term :math:`-k^2 u` takes very large values, making the equations extremely stiff.

Therefore, we need solvers that can handle veru stiff (constant) linear terms.

Available solvers
#################

ETD35 (default solver)
**********************

The default solver, :class:`ETD35 <pyloggrid.LogGrid.Framework.ETD35>`, is a wrapper around the ETD35 method by Whalen & al. from the `rkstiff <https://github.com/whalenpt/rkstiff>`_ package.

ETD4RK
******

:class:`ETD4RK <pyloggrid.LogGrid.Framework.ETD4RK>` is an implementation of ETD4RK from `Exponential time differencing for stiff systems (Cox & Matthews 2002) <https://doi.org/10.1006/jcph.2002.6995>`_ .

ViscDopri
*********

:class:`ViscDopri <pyloggrid.LogGrid.Framework.ViscDopri>` is our in-house old solver. It first solves the nonlinear equation using a reimplementation of scipy's DOPRI5 solver, then adds the linear term by either first-order `BDE <https://en.wikipedia.org/wiki/Backward_Euler_method>`_ or viscous splitting (depending on the value of `exact_viscous_splitting` in `solver_params`).

.. note:: **Legacy solvers**

    Current solvers accept a nonlinear and a linear term, and then internally compute the required timestep and updated fields.
    This workflow is not adapted for some specific schemes, such as the `Reversible Navier-Stokes equation <https://doi.org/10.1103/PhysRevE.107.065106>`, in which the viscosity at each step is adapted to keep viscosity constant.

    The current workaround is to use an old version of the PyLogGrid framework, which exposes the timestep in the linear step. Contact us directly if you need this version - in the future, we hope to be able to expose the new solvers in a better way.



