************
Introduction
************

Accurate numerical simulations of fluid dynamics usually involve Direct Numerical Simulations (DNS), whose numerical complexity increases as :math:`(L_{\max}/L_{\min})^3`, which becomes prohibitively slow for large inertial ranges.

Sparse models, such as shell models or REWA, alleviate this issue by only simulating a subset of the degrees of freedom of the system.

Logarithmic grids, or :doi:`log-lattices<10.1088/1361-6544/abef73>`  [1]_, are a sparse method for simulating fluid equations in N-D, while retaining the main symmetries of the underlying mathematical operators.

Pyloggrid is a Python library to perform and analyze simulations on log-lattices.

.. [1] Campolina, C. S., & Mailybaev, A. A. (2021). Fluid dynamics on logarithmic lattices. Nonlinearity, 34(7), 4684. doi:10.1088/1361-6544/abef73




