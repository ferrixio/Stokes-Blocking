# Stokes-Blocking
Repository of the spectral analysis of the Stokes equations with blocking preconditioning

Authors: Samuele Ferri, Chiara Giraudo, Valerio Loi, Miroslav Kuchta, Stefano Serra-Capizzano

## Introduction

This repository contains the material used to perform numerical tests of the blocking strategy [1] applied to the matrix sequences arising from the Taylor-Hood $\mathbb{P}_2-\mathbb{P}_1$ approximation of variable viscosity for 2$D$ Stokes problem under weak assumptions on the regularity of the diffusion [2].

### Index of the repository

- `matrices`: contains the colored output of the matrices in .xlsx;
- `src`: main folder of the methods. `StokesBuilder.py` is the true main file;
- `tests`: contains all the tests: eigenvalues adherences, clustering and PGMRES performance;

Python libraries used: numpy, random, math, matplolib, scipy.


## Related articles

[1] N. Barakitis, M. Donatelli, S. Ferri, V. Loi, S. Serra-Capizzano, R. Sormani. *Blocking structures, approximation, and preconditioning*. Numerical Algorithms. [10.1007/s11075-025-02157-y](https://link.springer.com/article/10.1007/s11075-025-02157-y). Online 8-7-2025.

[2] S. Ferri, C. Giraudo, V. Loi, M. Kuchta, S. Serra-Capizzano. *Spectral analysis of the stiffness matrix sequence in the approximated Stokes equation*. [Arxive preprint](https://doi.org/10.48550/arXiv.2510.25252). 2025.