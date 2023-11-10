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

Accurate simulations of fluid dynamics, such as Direct Numerical Simulations (DNS), become prohibitevely costly as we increase the inertial range. This is in particular a problem in simulating either the dynamics of singularities, or that of geophysical and astrophysical systems. Sparse simulation models, such as shell models [@gloaguen85; @biferale03] or REWA [@grossmann96], offer a cost-effective way to simulate such equations, by only considering a subset of the degrees of freedom, but at the cost of physical fidelity. Log-lattices [@martins_fluid_2019; @martins_fluid_2022] are a sparse model which conserves symmetries of the mathematical operators in a better way than previous methods.

# Statement of need

A minimal Matlab framework by Campolina & al [@campolina2020loglatt]. already exists, but other than relying on proprietary software, its capabilities are limited. PyLogGrid was designed to offer a solid, open-source, and extensive framework to perform log-lattice simulations. It enables both simulation, analysis and vosualisation tools. The choice of Python+C offers both great flexibility and speed.

PyLogGrid has been used in a number of publications [@barral2023asymptotic; @costa2023reversible].

