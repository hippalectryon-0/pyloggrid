*******
Context
*******

.. note:: This tutorial is based on our article :doi:`Asymptotic ultimate regime of homogeneous Rayleigh–Bénard convection on logarithmic lattices<10.1017/jfm.2023.204>`. For more information, refer to the paper.

Rayleigh-Bénard
***************

.. image:: https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/ConvectionCells.svg/1280px-ConvectionCells.svg.png
  :alt: <Rayleigh-Bénard illustration>
  :width: 400

`Rayleigh-Bénard equations <https://en.wikipedia.org/wiki/Rayleigh%E2%80%93B%C3%A9nard_convection>`_ describe the evolution of a volume of fluid heated from below in a container, and read in their basic form:

:math:`\nabla \cdot \mathbf{u} = 0`

:math:`\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\frac{1}{\rho} \nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{g} \alpha (T - T_0)`

:math:`\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T = \kappa \nabla^2 T`

where :math:`u` is the velocity, :math:`p` the pressure, :math:`\rho` the density, :math:`\nu` the kinematic viscosity, :math:`\mathbf{g}` the acceleration due to gravity, :math:`\alpha` the thermal expansion coefficient, :math:`T` the temperature, :math:`T_0` a reference temperature, :math:`\kappa` the thermal diffusivity.

(Homogeneous) Rayleigh-Bénard on Log-lattices
*********************************************

Standard Rayleigh-Bénard equations can't be translated as-is on Log-lattices, because:

1. the term :math:`\mathbf{g} \beta (T - T_0)` doesn't nicely convert to a finite Fourier expansion

2. the boundary conditions (heated from below, in a container) can't be represented in finite Fourier space either, which only accepts periodic boundary conditions.

Therefore, we instead simulate Homogeneous Rayleigh-Bénard (HRB), which corresponds to the following set of equations:

:math:`\nabla \cdot \mathbf{u} = 0`

:math:`\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\frac{1}{\rho} \nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{g} \alpha \theta \mathbf{z}`

:math:`\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T = \kappa \nabla^2 T + u_z\frac{\Delta T}{H}`

where :math:`\theta` is the temperature fluctuation relative to an affine profile, :math:`\Delta T` the mean imposed temperature gradient, :math:`H` the height of the container, :math:`\mathbf{z}` the vertical unit vector.

Using the Rayleigh number :math:`Ra=\alpha gH^3\Delta T/(\nu\kappa)` and the Prandl number :math:`Pr=\nu/\kappa`, those can be rewritten in adimentionalized form

:math:`\partial_t u_i = \mathbb{P}\left[-u_j\partial_ju_i+\frac{Pr}{Ra}\nabla^2u_i-fu_i\delta_{k\approx k_{min}}\right]_i`

:math:`\partial_t\theta_i = -u_i\partial_i\theta + u_z + \frac{\nabla^2\theta}{\sqrt{RaPr}}-f\delta_{k\approx k_{min}}`

where :math:`\mathbb{P}(A)=A-(k_i/k^2)k_jA_j` is a projector that accounts for the pressure term under divergence-free conditions, and :math:`f\delta_{k\approx k_{min}}` is an additional friction term that suppresses exponential instabilities (read the article for more detail and justification).
