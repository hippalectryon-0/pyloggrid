**************
Implementation
**************

Full simulation file
********************

:download:`HRB.py </static/code/HRB.py>`

Step-by-step
************

As in :doc:`../tuto_basic/tutorial_first_simu`, we create a new file ``HRB.py`` in ``Simulations/``.

We then implement the HRB equations:

Non-linear term
###############

This corresponds to :math:`\partial_t u_i = \mathbb{P}\left[-u_j\partial_ju_i\right]_i, \partial_t\theta_i = -u_i\partial_i\theta + u_z`

.. code-block:: python

    def equation_nonlinear(_t: float, grid: Grid, _simu_params: dict) -> dict[str, np.ndarray]:
        M = grid.maths
        ux, uy, uz, theta = grid.field("ux", "uy", "uz", "theta")

        # Convolutions
        uxdxux, uydyux, uzdzux, uxdxuy, uydyuy, uzdzuy, uxdxuz, uydyuz, uzdzuz, uxdxtheta, uydytheta, uzdztheta = M.convolve_batch(
            (
                (ux, M.dx * ux),
                (uy, M.dy * ux),
                (uz, M.dz * ux),
                (ux, M.dx * uy),
                (uy, M.dy * uy),
                (uz, M.dz * uy),
                (ux, M.dx * uz),
                (uy, M.dy * uz),
                (uz, M.dz * uz),
                (ux, M.dx * theta),
                (uy, M.dy * theta),
                (uz, M.dz * theta),
            )
        )

        # w/o pressure
        dux_dt = -uxdxux - uydyux - uzdzux
        duy_dt = -uxdxuy - uydyuy - uzdzuy
        duz_dt = -uxdxuz - uydyuz - uzdzuz + theta
        dtheta_dt = -uxdxtheta - uydytheta - uzdztheta + uz

        # Add pressure
        dux_dt, duy_dt, duz_dt = grid.maths.P_projector([dux_dt, duy_dt, duz_dt])

        return {"ux": dux_dt, "uy": duy_dt, "uz": duz_dt, "theta": dtheta_dt}

Linear term
###########

We then add the linear viscous/friction part :math:`\partial_t u_i = \mathbb{P}\left[\frac{Pr}{Ra}\nabla^2u_i-fu_i\delta_{k\approx k_{min}}\right]_i, \partial_t\theta_i = \frac{\nabla^2\theta}{\sqrt{RaPr}}-f\delta_{k\approx k_{min}}`:

.. code-block:: python

    def equation_linear(_t: float, grid: Grid, simu_params: dict) -> dict[str, np.ndarray]:
        M = grid.maths
        Pr, Ra = simu_params["Pr"], simu_params["Ra"]

        f = simu_params["f"] * np.ones_like(grid.ks_modulus)
        f[grid.ks_modulus > grid.k_min * grid.l ** 3] = 0

        visc = np.sqrt(Pr / Ra) * M.laplacian - f
        visc_theta = 1 / np.sqrt(Ra * Pr) * M.laplacian - f

        return {"ux": visc, "uy": visc, "uz": visc, "theta": visc_theta}

Initial conditions
##################

We initilize fields with random values at large scales:

.. code-block:: python

    def initial_conditions(fields: dict[str, np.ndarray], grid: Grid, _simu_params: dict) -> dict[str, np.ndarray]:
        grid = grid.to_new_size_empty(N_points)
        ks = grid.ks_modulus

        wx = np.zeros_like(ks, dtype="complex")
        wy = np.zeros_like(ks, dtype="complex")
        wz = np.zeros_like(ks, dtype="complex")
        theta = randcomplex_like(ks) * np.ones_like(ks, dtype="complex") / 100
        theta[ks > grid.k_min * 4] = 0
        ux, uy, uz = grid.maths.rot3D_inv([wx, wy, wz])

        fields["theta"] = theta
        fields["ux"] = ux
        fields["uy"] = uy
        fields["uz"] = uz

        return fields

Grid size update
################

We update the grid size based on the fraction of energy contained in the outermost layers:

.. code-block:: python

    def update_gridsize(grid: Grid) -> int | None:
        """update the grid size based on the fraction of energy contained in the outermost layers"""
        E = grid.physics.energy()
        ux, uy, uz, theta = grid.field("ux", "uy", "uz", "theta")
        mask = grid.ks_modulus > grid.k_min * grid.l ** (grid.N_points - 1)
        # grid
        comp = np.max(np.abs(ux[mask]) + np.abs(uy[mask]) + np.abs(uz[mask]) + np.abs(theta[mask]))
        if comp / np.sqrt(E) > 1e-100:
            return grid.N_points + 1
        if comp / np.sqrt(E) < 1e-170 and grid.N_points > 5:
            return grid.N_points - 1

Core simulation code
####################

We then call the solver with a given set of parameters:

.. code-block:: python

    Pr = 1
    Ra = 1e12
    print(f"Chosen parameters Pr={Pr}, Ra={Ra}")
    fields = ["ux", "uy", "uz", "theta"]
    D = 3
    l_params = {"plastic": False, "a": 1, "b": 2}
    n_threads_convolution = 6
    simu_params = {"Pr": Pr, "Ra": Ra, "f": 1}
    N_points = 13
    rtol = 1e-1

    save_path = "results/saveRB"
    loadfromsave = False

    solver = Solver(
        fields_names=fields,
        equation_nl=equation_nonlinear,
        D=D,
        equation_l=equation_linear,
        l_params=l_params,
        simu_params=simu_params,
        n_threads=n_threads_convolution,
    )
    solver.solve(
        save_path=save_path,
        initial_conditions=initial_conditions,
        save_one_in=100,
        update_gridsize_cb=update_gridsize,
        solver_params={"rtol": rtol, "adapt_cutoff": 1e-5},
    )
