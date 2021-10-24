[![conda-forge version](https://anaconda.org/conda-forge/pypardiso/badges/version.svg)](https://anaconda.org/conda-forge/pypardiso) [![PyPI version](https://badge.fury.io/py/pypardiso.svg)](https://pypi.org/project/pypardiso/) [![pypardiso-tests](https://github.com/haasad/PyPardisoProject/actions/workflows/tests.yaml/badge.svg?branch=master)](https://github.com/haasad/PyPardisoProject/actions/workflows/tests.yaml)

# PyPardiso

PyPardiso is a python package to solve large sparse linear systems of equations with the [Intel oneAPI Math Kernel Library PARDISO solver](https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-fortran/top/sparse-solver-routines/onemkl-pardiso-parallel-direct-sparse-solver-iface.html), a shared-memory multiprocessing parallel direct sparse solver.

PyPardiso provides the same functionality as SciPy's [scipy.sparse.linalg.spsolve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html#scipy.sparse.linalg.spsolve) for solving the sparse linear system `Ax=b`. But in many cases it is significantly faster than SciPy's built-in single-threaded SuperLU solver.

PyPardiso is not a python interface to the PARDISO Solver from the [PARDISO 7.2 Solver Project](https://www.pardiso-project.org/) and it also doesn't currently support complex numbers. Check out [JuliaSparse/Pardiso.jl](https://github.com/JuliaSparse/Pardiso.jl/) for these more advanced use cases.

## Installation
[![Anaconda-Server Badge](https://anaconda.org/haasad/pypardiso/badges/installer/conda.svg)](https://conda.anaconda.org/haasad)

Use PyPardiso with the [anaconda](https://www.continuum.io/downloads) python distribution (use [miniconda](http://conda.pydata.org/miniconda.html) if you need to install it). PyPardiso makes use of the Intel Math Kernel Library that is [included for free with conda](https://www.continuum.io/blog/developer-blog/anaconda-25-release-now-mkl-optimizations) and therefore doesn't work with other distributions (at least for the moment).

To install PyPardiso:
```
conda install -c haasad pypardiso
```

## Basic usage
PyPardiso provides a `spsolve` and a `factorized` method that are significantly faster than their counterparts in [scipy.sparse.linalg](https://docs.scipy.org/doc/scipy-0.18.1/reference/sparse.linalg.html).
```
>>> from pypardiso import spsolve
>>> x = spsolve(A,b)
```
