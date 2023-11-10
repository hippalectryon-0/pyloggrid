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
affiliations:
 - name: Université Paris-Saclay, CEA, SPEC, 91191 Gif-sur-Yvette, France
   index: 1
 - name: Université Paris-Saclay, CEA, CNRS, SPEC, 91191 Gif-sur-Yvette, France
   index: 2
date: 11 November 2023
bibliography: paper.bib
---

# Summary

PyLogGrid is a framework to perform and analyze log-lattice simulations, as introduced by @martins_fluid_2019.

Accurate simulations of fluid dynamics, such as Direct Numerical Simulations (DNS), become prohibitevely costly as we increase the inertial range. This is in particular a problem in simulating either the dynamics of singularities, or that of geophysical and astrophysical systems. Sparse simulation models, such as shell models [@gloaguen85; @biferale03] or REWA [@grossmann96], offer a cost-effective way to simulate such equations, by only considering a subset of the degrees of freedom, but at the cost of physical fidelity. Log-lattices [@martins_fluid_2019; @martins_fluid_2022] are a sparse model which conserves symmetries of the mathematical operators in a better way than previous methods.

# Statement of need

A minimal Matlab framework by @campolina2020loglatt. already exists, but other than relying on proprietary software, its capabilities are limited. PyLogGrid was designed to offer a solid, open-source, and extensive framework to perform log-lattice simulations. It enables both simulation, analysis and visualisation of log-lattice data. The choice of Python+C offers both great flexibility and speed. PyLogGrid offers sigificantly more options than @campolina2020loglatt, including several solvers, support for $k_i=0$ modes, failsafe simulations, optimized save formats, etc.

PyLogGrid has been used in a number of publications [@barral2023asymptotic; @costa2023reversible].

# Availability, usage and documentation

PyLogGrid can be installed via Pypi using `pip install pyloggrid`. Its documentation is hosted on [readthedocs](https://pyloggrid.readthedocs.io), and includes a tutorial.

# Acknowledgements

We thank A. Mailybaev, C. Campolina, G. Costa, Q. Pikeroen, A. Lopez for stimulating discussions and minor contributions.

# Funding

This work received funding from the Ecole Polytechnique, from ANR EXPLOIT, grant agreement no. ANR-16-CE06-0006-01 and ANR TILT grant agreement no. ANR-20-CE30-0035.

