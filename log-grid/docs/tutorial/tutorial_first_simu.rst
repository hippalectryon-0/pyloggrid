*********************
Your first simulation
*********************

PyLogGrid is designed with ease-of-use in mind, so that users spend less time coding and more time thinking about the physics.

Unlike other big frameworks, users don't provide a "simulation file" composed only of parameters, but instead provide a python script, which offers much more flexibility.

-------------------

To get started, let's look at the examples in the ``Simulations`` folder of the source (available on `Github <https://github.com/hippalectryon-0/pyloggrid/tree/main/log-grid>`_ ). Copy it and navigate inside:

.. code-block:: bash

   cd Simulations

In this folder, you will find a simple script called ``NS3D.py`` which simulates the 3D Navier-Stokes equations.

Let's look at what's inside.

Simulation functions
####################

At the top of the ``NS3D.py`` file, you will find several functions that define the equation being simulated:

* ``get_forcing`` returns the forcing field for a given simulation grid. This function is optional, but it's good practice to define the forcing field separately from the equation itself.
* ``equation_nonlinear`` returns the nonlinear part of the equation. If your equation is :math:`\partial_t u=A(u)−b\cdot u`, then this corresponds to :math:`A(u)`.
* ``equation_linear`` returns the linear factor of the equation :math:`−b`.
* ``initial_conditions`` defines the initial conditions for the simulation.
* ``update_gridsize`` specifies when to update the size of the simulation grid.

For an in-depth explanation on how to use those, read :class:`pyloggrid.LogGrid.Framework.Solver`.

Parameters
##########

After defining the simulation functions, we set the parameters for the simulation. First, we define the physical parameters of the equation:

.. code-block:: python

    fields = ["ux", "uy", "uz"]  # the scalar fields to simulate
    D = 3  # the dimension of the space
    l_params = {"plastic": False, "a": 1, "b": 2}  # the grid spacing's parameters
    Re_F = 1e3
    simu_params = {"Re_F": Re_F}  # scalar parameter of the simulation, passed to the equation

Next, we define the numerical parameters:

.. code-block:: python

    rtol = 1e-4  # relative tolerance of the solver
    n_threads_convolution = 4  # parallelization
    N_points = 6  # initial size of the grid

Finally, we define the output parameters:

.. code-block:: python

    save_path = f"results/save_3D_f0{f0:.2e}_ReF{Re_F:.2e}"  # save path
    end_simulation = {"t": 2000, "ode_step": 1e10}  # when to end the simulation
    save_one_in = 50  # save one step every N real steps

Running the simulation
######################

The simulation can be run as-is, but since we set ``end_simulation = {"t": 2000, "ode_step": 1e10}}`` it will run for a very long time [1]_.

If you want to end the simulation after a specified amount of time, you can change this parameter to:

.. code-block:: python

    end_simulation = {"elapsed_time": 120}

This will automatically end the simulation after 120 seconds (i.e., 2 minutes).

To run the simulation, simply execute the NS3D.py script:

.. code-block:: python

    python NS3D.py

Once the simulation is complete, you can analyze the output data.


**Tips**

* By default, :mod:`numpy` is multithreaded. If you want to disable this feature (typically because you want control over how many CPUs your simulation take, and because numpy operations take up a negligible time in the simulation), simply import it as ``from Libs.singlethread_numpy import np`` at the top of your simulation file (before it's imported by any other library).
* If your simulation includes a forcing term, it's a good practice to put it in a separate function, as done in the example. That way, you can reuse it in the initialization or in the treatment.
* There are a number of functions in :mod:`pyloggrid.Libs.datasci` to help manipulating arrays, in particular complex random arrays, which is very useful for creating peusorandom forcings consistent across grid size changes.

.. [1] In itself this is not an issue, as we can analyze the outputs even while the simulation is running, and stop it manually by killing the process when we are done.
