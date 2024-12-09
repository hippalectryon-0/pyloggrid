---
title: 'PyLogGrid: A Python package for fluid dynamics on logarithmic lattices'
tags:
    - Python
    - fluid dynamics
    - turbulence
    - logarithmic lattices
authors:
    - name: Amaury Barral
      orcid: 0000-0002-5857-1290
      affiliation: 1
    - name: Bérengère Dubrulle
      orcid: 0000-0002-3644-723X
      affiliation: 2
      corresponding: true
    - name: Guillaume Costa
      orcid: 0009-0008-9913-459X
      affiliation: 3
    - name: Quentin Pikeroen
      orcid: 0009-0006-0345-6339
      affiliation: 3
    - name: Adrien Lopez
      affiliation: 3
affiliations:
- name: LMD-IPSL, Sorbonne-Universités, 75005 Paris, France
  index: 1
- name: Université Paris-Saclay, CEA, CNRS, SPEC, 91191 Gif-sur-Yvette, France
  index: 2
- name: Université Paris-Saclay, CEA, SPEC, 91191 Gif-sur-Yvette, France
  index: 3


date: 11 November 2023
bibliography: paper.bib
---

_The authors contributed to this work in unequal proportions, with Amaury Barral taking the lead in the majority of the research, while the remaining authors made valuable but comparatively minor contributions._

# Summary

PyLogGrid is a framework for performing and analyzing log-lattice simulations, as introduced by @martins_fluid_2019.

Accurate fluid dynamics simulations, such as Direct Numerical Simulations (DNS), become prohibitively costly as we increase the inertial range. This is, in particular, a problem in simulating the dynamics of singularities or geophysical and astrophysical systems. Sparse simulation models, such as shell models [@gloaguen85; @biferale03] or REWA [@grossmann96], offer a cost-effective way to simulate such equations by only considering a subset of the degrees of freedom, but at the cost of physical fidelity. Log-lattices [@martins_fluid_2019; @martins_fluid_2022] are a sparse model which conserves symmetries of the mathematical operators better than previous methods.

# Statement of need

A minimal Matlab framework by @campolina2020loglatt already exists, but it relies on proprietary software, and its capabilities are limited. PyLogGrid was designed to offer a solid, open-source, and extensive framework for log-lattice simulations. It enables both simulation, analysis and visualisation of log-lattice data. The choice of Python+C offers both great flexibility and speed. PyLogGrid offers significantly more options than @campolina2020loglatt, including several solvers, support for $k_i=0$ modes, failsafe simulations, optimized save formats, tests and documentation, etc.

PyLogGrid has been used in a number of publications [@barral2023asymptotic; @costa2023reversible; @atmos14111690].

# Basic features

*This corresponds to version 2.2.1*.

The basics of PyLogGrid consists in a `Solver` class to simulate equations on log-lattices, and a `DataExplorer` class to visualize and analyze resulting data.

Solving equations uses [rkstiff](https://github.com/whalenpt/rkstiff) [@whalen2015exponential] by default. Convolutions are optimized in C, can be multithreaded, parallelized, and use AVX.
Simulations can be interrupted and resumed, and the grid size is adaptative.
Equations are easy to write as a number of mathematical operators are available through `pyloggrid.LogGrid.Grid.Math`.

Several libraries in `pyloggrid.Libs` provide helper functions for different use cases such as I/O, data science, and (interactive) plotting. Data visualization is also multithreaded.

# Availability and documentation

PyLogGrid can be installed via Pypi using `pip install pyloggrid`. Its documentation is hosted on [readthedocs](https://pyloggrid.readthedocs.io), and includes a tutorial.

# Acknowledgements

We thank A. Mailybaev, C. Campolina for stimulating discussions and minor contributions.

# Funding

This work received funding from the Ecole Polytechnique, from ANR EXPLOIT, grant agreement no. ANR-16-CE06-0006-01 and ANR TILT grant agreement no. ANR-20-CE30-0035.

# References
