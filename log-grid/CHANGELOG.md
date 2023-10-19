# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add `A11 - 3D rotation` to archive

### Changed

- Change license
- Change doc to *Sphinx*, and improve many docs
- Rename `plotLib.pad_array_2D` to `plotLib.rightpad_array_2D`

### Fixed

- Fix `pyyaml` version
- Fix normalization of `datasci.randcomplex` (was square instead of circular)

## [2.1.0]

### Added

- Auto-lint with pre-commit
- Add optional `fill` arg to `datasci.ragged_array_to_array`
- Add `HRB v7 edgevisc` to archive

### Changed

- Tests use pyling fixtures
- Return axes in `interactive_spectrum`
- Auto-rescale `interactive_3D_logplot_positive` axes for each step
- Rename `fields_to_linear` to `fields_to_1D` and `linear_to_fields` to `D1_to_fields` to avoid a confusion with the linear equation part

### Fixed

- Fix TimeTracker output when time<1ms
- Fix `RLIMIT_STACK` setting error

## [2.0.0]

### Added

- Add test coverage
- Add timers to evaluate the time spend reading/writing to disk, and convolving. Also measure time spent in DataExplorer.
- Add a `TimeTracker` class to `Libs.misc` to help with time-tracking
- Add loading initial conditions from a step in a hdf5 file using `initial_conditions=["file.h5", step]`
- Add a tool in `Utils` to find the fastest compiler flags for the convolver
- Add a script in `Utils` to install gcc-12 from source
- Add utests
- Add deploy stage to CI to view latest benchmark result, + artifact
- Add a generated doc page, replaces the wiki
- Add `Grid.to_new_size` and `Grid.to_new_size_empty`

### Changed

- Change the way spectra are normalized, now divided by the number of pts in each shell. This should straighten spectra.
- Change the saving format to hdf5.
- Rename `update_parameters` to `update_json_settings` in `Libs.IOLib`
- Change the repo's name from `Log Grid` to `PyLogGrid`
- Move the `initial_conditions` argument from `Solver()` to `solver.solve` and merge it with `loadfromsave`: the `loadfromsave`argument is removed, to resume the simulation set `initial_conditions="loadfromsave"`.
- Improve the convolution, now compiled from C instead of Cython, and with several major improvements. Should now be >2x faster.
- Improve the convolution's multithreading. As part of this effort, the way convolutions are distributed changes. One can still call `convolve` as before, but a new function `convolve_batch` that takes a several (f,g) and computes their convolution in parallel now can be used.
- When `initial_conditions="loadfromsave"`, automatically load `dt` from the latest timestep.
- Rename `equation_update` to `equation_l` and `viscosity_update` to `equation_nl` in all its implementations.
- Remove `N_points` from `Solver()` args, pass it with `initial_conditions` instead.
- Change `Solver(fields_name=)` to `Solver.fields_names` to align with `Grid`.
- Move several attributes shared between `Solver` and `Grid` to be only owned by `Grid`.
- Remove `fields` from the arguments of `equation_l` and `equation_nl`. They should now be accessed via `grid.field(...)`.
- Only recompute the missing fields in `DataExplorer` if `loadfromsave=True` instead of recomputing all the fields.
- Hardcode a (better) default size for `plotshowm(full=True)` when the backend is non-interactive.
- `initial_conditions="loadfromsave"` now updates `end_simulation, solver_params, simu_params` rather than loading them from the existing simulation
- Move `Libs` and `LogGrid` into the `pyloggrid` package
- Change docstrings from REST to google style for improved lisibility
- Switch dependency management from `requirements.txt` to `poetry`

### Fixed

- Increase the max recursion depth to avoid crashing simulations with a lot of grid size changes
- Fix the save directory of a few tests
- Fix a crash in windows related to the `resource` library
- Ending a simulation no longer shows a stack trace of `InterruptedError` if the grid size is adaptive.

### Removed

- Remove the `save_data` arg from `DataExplorer.display` (now always `True`)

## [1.8.0]

### Added

- Add default MR template
- Add LittlewoodP code in Examples
- Add support in DataExplorer for varying grid sizes in outputs
- Add support for "u" fields in `grid.physics.enstrophy`
- Add `HRB v5` to archive
- Add a new solver based on https://github.com/whalenpt/rkstiff and made it default
- Add an `ax` argument to `plotLib.scatter`, if `None` defaults to `plt`
- Add `plotLib.interactive_spectrum`
- Add `datasci.logmean`, `datasci.fit`
- Add `grid` parameter to `DataExplorer.load_step`
- Add new optional `solver` parameter to `Solver.solve` to chose among the different solvers
- Add `Examples/ETD4RK`
- Add `interactive_grid_imshow, interactive_grid_3Dslice, interactive_3D_logplot_positive` in `plotLib`
- Add optional `ax` argument to `plotLib.legends`
- Add `PlotFun` custom type in `DataExplorer`
- Add `HRBv6` to archive
- Add new generic bencharking util
- Add a `save` argument to `plot_interactive_XX` in `plotLib` to save the interactive graphs as videos

### Changed

- Change the default solver (see above)
- In `DataExplorer`, a field `t` is automatically stored when `perstep=True`
- Fix `singlethread_np` import in `DataExplorer`
- `Grid.field` now returns a single array when called with a single argument
- Move `atol, rtol, exact_viscous_splitting` to a dict `solver_params` which may take additional parameters
- The very first and last steps of the simulation are now always saved
- Rename `interactive_3D_logplot` to `interactive_3D_logplot_by_z`
- Update the structure of save files, in which the order the fields is no longer relevant. This is not backwards-compatible, but legacy support has been added to read old save files.
- `end_simulation` no longer defaults to `t=1` in `solver.solve`
- `datasci.mean` now returns the mean *and the variance*
- add sliders to `Examples/LittlewoodPaley/proto_gamma.py` and update some plots
- `datasci.mean` now returns the mean and the std
- default ̀end_simulation` parameters no longer include `{"t": 1}`

### Fixed

- Fix `plotLib.scatter` when supplying `linestyle`
- Fix `interactive_3D_logplot_by_z` not updating properly
- Fix saving duplicate steps on grid size update

## [1.7.1]

### Added

- Add a `-s` flag to the automatic installation scripts to skip the installation of the python executable
- Add an option `exact_viscous_splitting: bool` in the solver. If enabled, we solve the equation via "exact" viscous splitting, by multiplying dydt by exp(-visc * t). If the viscous term is big, this requires much smaller time steps, and is significantly less stable. Not recommended for use in everyday simulations, but more as a baseline for comparison.
- Add a function in `treat_3D.py` to plot the "theoretical" injection and dissipation of energy. Note that those formulas are only local derivatives, and therefore are expected to slightly differ from the RK estimates. In particuler, there might be a time shift of the order of `dt`. /!\ The injection and viscosity need to be hardcoded in the treatment.
- Add0 `grid.maths.div2d`
- Add utests for NS3D evolution of energy and div2d
- Add `interactive_3D_logplot_by_z` to `PlotLib` to plot 2D/3D data in 3D with interactive visualisation (z slice / time)

### Changed

- Remove a hard to see marker from the default cycler in PlotLib
- Update requirements
- Change behavior of `drawables` argument passed by DataExplorer: was `dict`, is now a function that can be called as `drawables(key1, key2, ...)̀`

### Fixed

- Fix the new default method for calculating the viscous term (was not on by default)
- Remove bad markers from default plotting cycle
- Fix a sign error in `NS3D.py`

## [1.7.0]

### Added

- Add `launch_treat_batch.sh` as a template to launch/treat many simulations at once
- Add QG 1&2layer(s) to Archive
- Add HRBv3 to the archive
- Add missing HRBv2 .ods data and pdfs to the archive
- Add Draft folder in Example
- Add spectrum investigation code in Draft
- Add benchmark_rust to Draft
- Add `sys.path.insert` to NS3D.py for compatibility when launched directly from python
- Add options to select which steps to treat, the number of threads, and whether to save treatment results in DataExplorer
- Add functions `cumulative_k`, `compute_by_shell` to Grid.Physics
- Add `enforce_grid_symmetry`, used after each ode step to make sure there's no drift
- Add `simu_params` to solver `initial_conditions`
- Add `complete_array_with_nans` in `Libs/datasci`
- Add automated setup scripts for linux & windows. Those scripts install python, create a venv, install the requirements, and compile the convolver. They can be run on already existing installations to update them.
- Add `tight` option to `pltshowm`
- Add a `scatter` function to PlotLib
- Add a `field(name)` method to `Grid`

### Changed

- Bump pyparsing to >=3 since the Cython bug seems to have been fixed in newer versions
- Update README to use python3.11
- Update convolving utests to be faster (probe smaller grids in 3D)
- Change the handling of absolute tolerance. Now, atol is *relative* to the max possible value of y.
- Change `ujson` to `orjson`. This de-facto removes the support for serializing np.nan, np.inf as strings (now serialized as `null`), but afaik it was not of any use.
- Cleanup requirements
- Update the way custom colormaps are handled (was broken by [this matplotlib commit](https://github.com/matplotlib/matplotlib/commit/0b6b385e6d44cb2bddc16b5a46e85680109fc268))
- Change default plot style (more article-like, automatic linestyle and marker, colorblind-friendly, perceptually uniform colormap)
- Refactor the ViscDopri solver:
    - Change the signature of the `visc_update`, which is now expected to return only the value of the viscous term for each field
    - Change the computation of the viscous term, which should be significantly more precise as long as the timestep is not too big
    - Minor fixes
    - Split `visc_update_lin` using the new `save_step` method

### Fixed

- Fix the way RK updated the derivative after the viscosity step. It is now recomputed every step. This changes the signature of the viscosity
  update, so update your code accordingly (doesn't require/return the derivative anymore)
- Fix spectrum calculation crash with k0=True
- Fix Basilisk test missing fonts
- Fix some edge cases in the ViscDopri solver

## [1.6.0]

### Added

- Decorators to simply run tests on many parameters (N, D, l_params, k0)
- Support for k=0 mode (set `grid.k0=True`)
- Test for convolutions
- Test for 3D Euler, which checks that the divergence is zero and the energy is conserved over 100 ode steps.
- Added timings (instead of just speed) to benchmarking
- Add rainbow desaturated colormap as default
- Add min and max solver step size
- Add "compact" option to pltshowm
- The latest convolution kernel is automatically cached, ensuring that if the grid doesn't change it's only computed once.
- Added `cumulative_k` function in Grid.Physics
- Added utests for k0 and many more parameter combinations

### Changed

- drastically changed the way convolutions are computed. We compute a convolution kernel on grid initialisation (which
  computes all the triads), then we simply iterate over this kernel. This results in a faster convolution (and it makes
  handling the k=0 mode easier).
- update typing to py3.9 (`typing.Dict -> dict`, ...)
- `solver.ode_step` no longer resets on a resumed run. It can now be used to set the end of the simulation.
- Changed `laplacian3D_inv` and `laplacian3D` to `laplacian_inv` and `laplacian` in `Grid.Math` since the formula is
  generic
- Change all `Grid.Math` linear operators to properties: `M.dx(arr)` becomes `M.dx * arr`. This allows to easily invert
  them, ex: `M.laplacian_inv * arr = M.inv(M.laplacian) * arr`
- Set default plot behavior to `initFormat`

### Fixed

- Fixed the formula of `P_projector`
- Added rough support for k0 mode in spectrum (not calculated for k=0, but won't error anymore)

## [1.5.0]

### Added

- Add `P_projector` in `Grid` (faster to use than going through the rotational)
- Add a `save` and `legend` option in `pltshowm`
- Add the property`grid.ks_1D_mid` to represent k values of data averaged over `ks_1D` intervals (ex spectras)
- Add `initFormat` function to `plotLib.py` to easily make plots that are formatted for scientific publications
- Add an `Examples` folder that showcases the following features from the past months (none of which are polished): 2D
  basilisk code, JP Laval's Fortran code and the visualisation of its .nc outputs, the new `initFormat` function.
- Add `random_cmap` in `Utils` to generate a random colormap with N items (from SO)

### Changed

- Set default number of CPUs used to max//2 instead of max (which hogs all the resources and slows down a lot). We still
  recommend setting the numbr of threads manually for best performances.
- Updated several libs in `requirements.txt`
- Moved old files to the Archive, restructured the Archive for a clearer chronological browsing
- Benchmarking now also tests running several convolutions at once using Joblib. The final graph has also been reworked.

### Fixed

- The end_simulation parameters now have proper default values
- Grid fields get updated earlier in `visc_update_lin` (was done later, could result in incorrect results)
- Fixed a sign in `P_projector`
- Fix singlethreading import in `NS3D.py`

## [1.4.0]

### Added

- Unit tests for `datasci.py`, `custom_logger.py`, `IOLib.py`, `misc.py`, `Grid.py` except for Grid.Maths.convolve
- Requirements: `deepdiff->5.2.3`
- function `randcomplex` in `datasci.py` to generate normal-random complex arrays from a shape

### Changed

- Requirements: numpy `1.20.1->1.20.2`
- Renamed `rot_inv_3D_divfree` to `rot3D_inv`, `rot_inv_3D` to `rot3D_inv` (same 2D), `laplacian_inv`
  to `laplacian3D_inv`
- `powerlaw_fit` can take an additional argument `a` to fit with a determined power law
- much faster `spectrum` method. Spectrum's callback now takes a bool mask of points as its secondary arg.

### Fixed

- `powerlaw_fit`'s `mask` argument now defaults to `None`
- `laplacian3D_inv` now has the correct sign
- `spectrum` was divided by a spurrious factor 2

### Removed

- Removed legacy support for `.npy` save files
- Removed `energyspectrum` now that we have the more general `spectrum`
- Removed `wkmax` (lack of utility)

## [1.3.0]

### Added

- Log means in `treat_3D.py` for Re and "epsilon" dissipation rates
- Install instructions in README
- grid real-space length and volume are accessible via properties of the Grid() object
- new function `plot2axesX` in `plotlib` to get a figure with two X axis

### Changed

- pltshowm now plots with a grid (alpha=0.3)

### Fixed

- HRB Scalings in Reynolds and dissipation rates (epsilons)

## [1.2.0]

### Added

- changelog

### Fixed

- Saved source now correctly saved as `.py`
- .gitignore now properly ignores pycache
- reflected previous namechanges in `launch_batch.sh`
- Libraries path added to simulation env. SImulation files can be run out-of-the-box without fiddling with the source
  dir
- `datasci.py` now properly singlethreads (was broken by a previous import of scipy)
- Python no longer errors if plotting on a backend that doesn't support fullscreen
- Missing rescaling time in several `treat_?D` functions

## [1.1.0] - 2021-03-25

### Added

- Simulation source files (the "topmost" file ran) is saved as `source.py` in the save dir

## [1.0.3_release] - 2021-03-25

Warning: this release is NOT backwards-compatible in any way.

Since it's the first one of the changelog, I will not bother listing all the changes from the previous (undocumented)
release.

### Added

- Proper gitlab CI (only lint for now)
- Licence CC BY-NC-SA 4.0

### Changed

- Whole project linted, inline-documented, refactored (also structure-wise)
- Numpy threads capped to 1
- Prints replaced by logging

### Fixed

- Corrupted JSON files on a brutal stoppage are now backed up. Writing process is more robust.
- Fix last step > max step in DataExplorer
- Too many fixes to list here. Later changelog entries will be more precise.
