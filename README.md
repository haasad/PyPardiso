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


## Changelog

__v0.3.2__

* Change requirements in setup.py to fix [failing conda-forge build](https://github.com/conda-forge/staged-recipes/pull/15106#discussion_r654696146)

__v0.3.1__

- Revert to the old way of detecting the mkl_rt library on osx, since psutil doesn't work (see [#14])

__v0.3.0__

- Changed how pypardiso detects the __mkl_rt__ library to fix a breaking change on windwos with [mkl 2021.2.0](https://anaconda.org/conda-forge/mkl). See [#12](https://github.com/haasad/PyPardisoProject/issues/12) for details.

__v0.2.2__

- CSR-matrix format is forced in `spsolve` and `factorized`. This fixes a serious compatibility issue with [brightway2](https://brightwaylca.org), where a technosphere matrix in CSC-format produces wrong results, due to the bad conditioning of the matrix (see details in issue #7).

__v0.2.1__

- Switched from zero- to one-based indexing for the call to the pardiso library. This brings performance of the factorization phase back to the level of v0.1.0, v0.2.0 is much slower.
