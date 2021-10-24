# PyPardisoProject
[![Anaconda-Server Badge](https://anaconda.org/haasad/pypardiso/badges/version.svg)](https://anaconda.org/haasad/pypardiso) [![PyPardiso-Tests](https://github.com/haasad/PyPardisoProject/actions/workflows/conda-pytest.yaml/badge.svg?branch=master)](https://github.com/haasad/PyPardisoProject/actions/workflows/conda-pytest.yaml)

Python interface to the [Intel MKL Pardiso library](https://software.intel.com/en-us/node/470282) to solve large sparse linear systems of equations

More documentation is coming soon. In the meantime, refer to the comments and docstrings in the source code.

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
