## Changelog

__v0.3.3__

* Release on PyPI and anaconda.org/haasad with github actions (see #19 and #20)

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
