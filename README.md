# Walk-on-Interfaces
This repository contains author's implementation for the 2026 JCP paper ["Walk-on-Interfaces: A Monte Carlo Estimator for an Elliptic Interface Problem with Nonhomogeneous Flux Jump Conditions and a Neumann Boundary Condition"](https://www.sciencedirect.com/science/article/pii/S0021999126004584?__cf_chl_tk=d5zJCnbDx74t6jaN6Nheya0JkgZtG4fYb2D6rrAQPyo-1780676013-1.0.1.1-kfKG1D8HYH_ARrvou55gkguARzXl7TuTgt1B3vxETmk), by Xinwen Ding and Adam Stinchcombe.

## Overview
We study numerical methods for elliptic PDEs of the form

```math
-\nabla \cdot (\sigma(x) \nabla u(x)) = 0, \quad x \in \Omega
```
with discontinuous coefficienet $\sigma(x)$. A Monte Carlo method is propsed and developed here. 

This repository includes:

- Data/domain sampling routines
- Implementation of our algorithm
- Examples we have provide in the paper

## Repository Structure

The structure roughly looks like the follows:
```text
.
├── woi2D/                 # WoI for 2D domains
│   ├── src                # WoI core implementation
│   ├── lib                # AABBTree
│   ├── main_ex1.m         
│   └── main_ex3_2D.m 
├── woi3D/                 # WoI for 3D domains
|   ├── src                # WoI core implementation
│   ├── lib                # AABBTree
│   ├── main_ex3_3D.m  
│   ├── main_ex4.m
|   └── main_ex5.m
├── woi_high_dim/          # WoI for higher dimensional domains
|   ├── src                # WoI core implementation
│   ├── main3D.m
|   ├── main4D.m
|   ├── main5D.m
|   └── main6D.m
├── OBJ2D/                  # .obj file for 2D shapes
├── OBJ3D/                  # .obj file for 3D shapes
└── README.md

```

## Installation
```bash
git clone https://github.com/XinwenDing/Walk-on-Interfaces.git
```

## Dependency
* [gptoolbox](https://github.com/alecjacobson/gptoolbox)
* [matlab-tree](https://github.com/tinevez/matlab-tree)

## Quick Start
Any file starts with main is a good starting point.

## Citation
```bibtex
@article{DING2026115105,
title = {Walk-on-Interfaces: A Monte Carlo Estimator for an Elliptic Interface Problem with Nonhomogeneous Flux Jump Conditions and a Neumann Boundary Condition},
journal = {Journal of Computational Physics},
pages = {115105},
year = {2026},
issn = {0021-9991},
doi = {https://doi.org/10.1016/j.jcp.2026.115105},
url = {https://www.sciencedirect.com/science/article/pii/S0021999126004584},
author = {Xinwen Ding and Adam R Stinchcombe},
keywords = {Elliptic interface problem, Integral equations, Monte Carlo methods, Scientific machine learning}
}
```