*****************
Saved data format
*****************

Several kinds of data are saved as one uses the Framework: simulation outputs, simulation settings, treatment results, etc.

Current format
##############

Simulation output
*****************

All the outputs of the simulation are saved in a single ``output.h5`` hdf5 file using h5py. Each step ``N`` of the simulation corresponds to a group ``step_N``, and each field ``ux, uy, ...`` is saved as one dataset with the same name in the group.

The following data is saved in each step (group ``step_N``):

* As attributes:

    * ``t``: the simulation's time
    * ``N_points``: the base size of the grid
    * ``k_min``: the grid's minimum wavevector
    * ``k0``: if True, the grid supports the k=0k=0 mode, false otherwise
    * ``ttrack_X``: elapsed time for key X

* As datasets

    * The fields on the grid, each named after their field name.

Simulation settings
*******************

The settings of the simulation are saved in a ``settings.json`` JSON file using orjson. Note that this library doesn't allow ``np.nan``, ``np.inf``-like objects. We do not put this inside the ``output.h5`` file for two reasons. First, the data is too unstructured (nested dictionaries of arbitrary types). Second, we want the settings to be human-readable.

The following data is saved in the settings:

* ``init_t``: the time at which to start the simulation
* ``end_simulation={"t", "elapsed_time", "step", "ode_step"}``

    * ``t``: time at which to end the simulation
    * ``elapsed_time``: real-life time elapsed at which to end the simulation
    * ``step``: saved step N at which to end the simulation
    * ``ode_step``: number of ode steps at which to end the simulation

* ``N_steps``: number of saved sters since the start
* ``N_ode_steps``: number of performed ode steps since the start
* ``l_params, D, fields_name, simu_params``: same as set in Solver()
* ``solver_params``: same as set in Solver.solve

Simulation source
*****************

The source used to run the simulation is saved as ``source.py``. The goal is to remember a long time after we have run the simulation what was the exact code used (forcing, initial conditions, ...). For the same human-readability reason as above, we don't put it in ``output.h5``.

Treatment computations
**********************

The computed quantities from the treatment of the simulation are saved in ``drawables.npy``. The goal is to be able to reuse them without recomputing if we want to plot the same data a different way. As they also contain nested structures of arbitrary length, it is unpractical to save them in ``output.h5``.

Treatment outputs
*****************

Most treatment functions save an image of their plot.

Backups
*******

Both ``output.h5`` and ``settings.json`` are frequently written to. As a result, there is a significant risk of data corruption if the process is killed while writing. To avoid this risk, a backup of each file ``output_bk.h5`` and ``settings_bk.json`` is saved. Once the main file has been written to and closed, we update the backup. When reading either file, if we fail to read the main one, we fall back on the backup.

*Downside*: Now that we save all the outputs as a big file, the backup file takes a significant space, effectively doubling the disk space of the simulations. The solution is to prune the backup files once the simulation is finished. A python tool to automatically prune a directory recursively is planned in issue #31.

Old format (9de8071cd20936f8a2c0839f91fa63272eff66aa)
*****************************************************

*This save format is still supported in 2.x, but may be discontinued in a future version*

Each time step is saved in a separate ``.npz`` file. Each file contains a dict with keys ``{fields, t, N_points, k_min, elapsed_time, k0, ...}`` (same meaning as above).

*Downside*: This creates a lot of small files, which is a pain when moving, deleting, listing etc.

Older format (86a41d9d99e26b2c3bebcce7ff15c2ccb0f521c5)
*******************************************************

*This save format is still supported in 2.x, but may be discontinued in a future version*

Each time step is saved in a separate ``.npz`` file. Each file contains a dict with keys ``{arr_0, arr_1, ...}`` which are then remapped onto ``{fields, t, N_points, k_min, elapsed_time, k0, ...}`` (same meaning as above).

Downside: The load/save order of the arrays is important. We can't add arbitrary data to the save.
