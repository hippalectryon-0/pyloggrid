********************
Analyzing the output
********************

After running ``NS3D.py``, you can easily analyze its output using ``treat_3D.py``.

Analysis settings
#################

Begin by specifying which analysis functions you want to run. In treat_3D.py, at the end of the file, you can toggle analysis for different functions by (un)commenting the corresponding lines. For instance:

.. code-block:: python

    draw_funcs = {
        # "simu_params": {"get": get_simu_params, "plot": plot_simu_params},
        "spectrum_and_energy": {"get": get_spectrum_and_energy, "plot": plot_spectrum_and_energy},
        # "spectrum": {"get": get_spectrum, "plot": plot_spectrum, "perframe": False},
        # "bxyz": {"get": get_bxyz, "plot": plot_bxyz},
        # "3D": {"get": get_3D, "plot": plot_3D, "perframe": False},
        "epsilon": {"get": get_epsilon, "plot": plot_epsilon},
        # "uRMS": {"get": get_uRMS, "plot": plot_uRMS, "perframe": False},
    }

In this example, only the ``spectrum_and_energy`` and ``epsilon`` functions are active. You can toggle any function by commenting or uncommenting the corresponding line.

Next, you need to set the parameters for the analysis:

.. code-block:: python

    save_path = f"results/save_3D_f0{f0:.2e}_ReF{Re_F:.2e}"  # where the simulation data is saved
    temp_path = "results/temp"  # temp path to store images before compiling to video
    N_points = 500  # how many time points max to load
    n_jobs = 3  # parallelization
    loadfromsave = False  # load already computed results /!\ if your data has changed, this may show the old data

Treatment functions
###################

Functions that can be computed step by step
*******************************************

You can create your own treatment function to plot a custom quantity as a function of time.

To do this, you need to create two functions: one to compute the energy ``get_custom``, and another to plot the results ``plot_custom``.

Here's an example of how to create the ``get_custom`` function:

.. code-block:: python

    def get_custom(grid: Grid, t: float, _simu_params: dict) -> dict:
        """
        A custom treatment function. Computes the energy (x2) in ux, and a custom quantity.
        """

        ux, uy, uz = grid.fields("ux", "uy", "uz")  # retrieve the fields at time t
        qty1 = grid.maths.self_inner_product(ux)
        qty2 = uy[0] * uz[0] * t

        return {"qty1": qty1, "qty2": qty2}  # return the computed quantities

And here's an example of how to create the corresponding ``plot_custom`` function:

.. code-block:: python

    def plot_custom(drawables: PlotFun) -> None:
        from Libs.plotLib import pltshowm, labels, scatter
        ts, qty1, qty2 = drawables("t", "qty1", "qty2")  # retrieve t and the computed quantities
        _, ax = plt.subplots()

        scatter(ax, ts, qty1, label="$E_x$")
        scatter(ax, ts, qty2 * qty1[-1] / qty2[-1], label="custom qty")
        labels("t", "", "My custom graph")
        pltshowm()

To execute your custom function, add it to the ``draw_funcs`` dictionary:

.. code-block:: python

    draw_funcs = {
        # ... other functions,
        "custom": {"get": get_custom, "plot": plot_custom},
    }

When you run the script

.. code-block:: bash

    python treat_3D.py

you should see a graph like this

.. image:: /static/img/tutorial/tutorial_graph.png
  :alt: <image: Simulation treatment graph>

Arbitrary functions
*******************

Arbitrary functions allow you to perform custom analysis that spans multiple time steps, such as averaging or integrating over time. To use arbitrary functions, add ``"perframe": False`` to the function's settings in the ``draw_funcs`` dictionary:

.. code-block:: python

    draw_funcs = {
        # ... other functions,
        "custom": {"get": get_custom, "plot": plot_custom, "perframe": False},
    }

The get_custom function should now take a :class:`pyloggrid.LogGrid.DataExplorer` object as input, which allows you to load data from any time step:

.. code-block:: python

    def get_custom(dexp: DataExplorer) -> dict:
        """
        A custom treatment function that requires to load multiple time steps
        """

        # For example if we need to average over the last 10 steps:
        N_avg = 10
        qty = []
        for curr_step in range(max(dexp.N_steps - N_avg, 1), step + 1):
            _, grid = dexp.load_step(curr_step)
            qty.append(<your custom analysis code here>)

        return {"some_data": np.mean(qty)}  # return the computed quantities

The ``plot_custom`` function for arbitrary functions is similar to that of ``"perframe": True`` functions, with the only difference being that the ``t`` parameter is not automatically added to the drawables object.
