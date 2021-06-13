# coding: utf-8
import os
import sys
import ctypes
import warnings
import hashlib

import psutil
import mkl
import numpy as np
import scipy.sparse as sp
from scipy.sparse import SparseEfficiencyWarning


class PyPardisoSolver:
    """
    Python interface to the Intel MKL PARDISO library for solving large sparse linear systems of equations Ax=b.

    Pardiso documentation: https://software.intel.com/en-us/node/470282

    --- Basic usage ---
    matrix type: real (float64) and nonsymetric
    methods: solve, factorize

    - use the "solve(A,b)" method to solve Ax=b for x, where A is a sparse CSR (or CSC) matrix and b is a numpy array
    - use the "factorize(A)" method first, if you intend to solve the system more than once for different right-hand
      sides, the factorization will be reused automatically afterwards


    --- Advanced usage ---
    methods: get_iparm, get_iparms, set_iparm, set_matrix_type, set_phase

    - additional options can be accessed by setting the iparms (see Pardiso documentation for description)
    - other matrix types can be chosen with the "set_matrix_type" method. complex matrix types are currently not
      supported. pypardiso is only teste for mtype=11 (real and nonsymetric)
    - the solving phases can be set with the "set_phase" method
    - The out-of-core (OOC) solver either fails or crashes my computer, be careful with iparm[60]


    --- Statistical info ---
    methods: set_statistical_info_on, set_statistical_info_off

    - the Pardiso solver writes statistical info to the C stdout if desired
    - if you use pypardiso from within a jupyter notebook you can turn the statistical info on and capture the output
      real-time by wrapping your call to "solve" with wurlitzer.sys_pipes() (https://github.com/minrk/wurlitzer,
      https://pypi.python.org/pypi/wurlitzer/)
    - wurlitzer dosen't work on windows, info appears in notebook server console window if used from jupyter notebook


    --- Number of threads ---
    methods: get_max_threads, set_num_threads
    - you can control the number of threads by using the "set_num_threads" method


    --- Memory usage ---
    methods: remove_stored_factorization, free_memory

    - remove_stored_factorization can be used to delete the wrapper's copy of matrix A
    - free_memory releases the internal memory of the solver

    """

    def __init__(self, mtype=11, phase=13, size_limit_storage=5e7):

        if sys.platform == 'darwin':
            self.libmkl = ctypes.CDLL('libmkl_rt.dylib')
        else:
            # find the correct mkl_rt library by searching the loaded libraries of the process
            proc = psutil.Process(os.getpid())
            mkl_rt = [lib.path for lib in proc.memory_maps() if 'mkl_rt' in lib.path][0]
            self.libmkl = ctypes.CDLL(mkl_rt)

        self._mkl_pardiso = self.libmkl.pardiso

        # determine 32bit or 64bit architecture
        if ctypes.sizeof(ctypes.c_void_p) == 8:
            self._pt_type = (ctypes.c_int64, np.int64)
        else:
            self._pt_type = (ctypes.c_int32, np.int32)

        self._mkl_pardiso.argtypes = [ctypes.POINTER(self._pt_type[0]),    # pt
                                      ctypes.POINTER(ctypes.c_int32),      # maxfct
                                      ctypes.POINTER(ctypes.c_int32),      # mnum
                                      ctypes.POINTER(ctypes.c_int32),      # mtype
                                      ctypes.POINTER(ctypes.c_int32),      # phase
                                      ctypes.POINTER(ctypes.c_int32),      # n
                                      ctypes.POINTER(None),                # a
                                      ctypes.POINTER(ctypes.c_int32),      # ia
                                      ctypes.POINTER(ctypes.c_int32),      # ja
                                      ctypes.POINTER(ctypes.c_int32),      # perm
                                      ctypes.POINTER(ctypes.c_int32),      # nrhs
                                      ctypes.POINTER(ctypes.c_int32),      # iparm
                                      ctypes.POINTER(ctypes.c_int32),      # msglvl
                                      ctypes.POINTER(None),                # b
                                      ctypes.POINTER(None),                # x
                                      ctypes.POINTER(ctypes.c_int32)]      # error

        self._mkl_pardiso.restype = None

        self.pt = np.zeros(64, dtype=self._pt_type[1])
        self.iparm = np.zeros(64, dtype=np.int32)
        self.perm = np.zeros(0, dtype=np.int32)

        self.mtype = mtype
        self.phase = phase
        self.msglvl = False

        self.factorized_A = sp.csr_matrix((0, 0))
        self.size_limit_storage = size_limit_storage
        self._solve_transposed = False

    def factorize(self, A):
        """
        Factorize the matrix A, the factorization will automatically be used if the same matrix A is passed to the
        solve method. This will drastically increase the speed of solve, if solve is called more than once for the
        same matrix A

        --- Parameters ---
        A: sparse square CSR matrix (scipy.sparse.csr.csr_matrix), CSC matrix also possible
        """

        self._check_A(A)

        if A.nnz > self.size_limit_storage:
            self.factorized_A = self._hash_csr_matrix(A)
        else:
            self.factorized_A = A.copy()

        self.set_phase(12)
        b = np.zeros((A.shape[0], 1))
        self._call_pardiso(A, b)

    def solve(self, A, b):
        """
        solve Ax=b for x

        --- Parameters ---
        A: sparse square CSR matrix (scipy.sparse.csr.csr_matrix), CSC matrix also possible
        b: numpy ndarray
           right-hand side(s), b.shape[0] needs to be the same as A.shape[0]

        --- Returns ---
        x: numpy ndarray
           solution of the system of linear equations, same shape as input b
        """

        self._check_A(A)
        b = self._check_b(A, b)

        if self._is_already_factorized(A):
            self.set_phase(33)
        else:
            self.set_phase(13)

        x = self._call_pardiso(A, b)

        # it is possible to call the solver with empty columns, but computationally expensive to check this
        # beforehand, therefore only the result is checked for infinite elements.
        # if not np.isfinite(x).all():
        #    warnings.warn('The result contains infinite elements. Make sure that matrix A contains no empty columns.',
        #                  PyPardisoWarning)
        # --> this check doesn't work consistently, maybe add an advanced input check method for A

        return x

    def _is_already_factorized(self, A):
        if type(self.factorized_A) == str:
            return self._hash_csr_matrix(A) == self.factorized_A
        else:
            return self._csr_matrix_equal(A, self.factorized_A)

    def _csr_matrix_equal(self, a1, a2):
        return all((np.array_equal(a1.indptr, a2.indptr),
                    np.array_equal(a1.indices, a2.indices),
                    np.array_equal(a1.data, a2.data)))

    def _hash_csr_matrix(self, matrix):
        return (hashlib.sha1(matrix.indices).hexdigest() +
                hashlib.sha1(matrix.indptr).hexdigest() +
                hashlib.sha1(matrix.data).hexdigest())

    def _check_A(self, A):
        if A.shape[0] != A.shape[1]:
            raise ValueError('Matrix A needs to be square, but has shape: {}'.format(A.shape))

        if sp.isspmatrix_csr(A):
            self._solve_transposed = False
            self.set_iparm(12, 0)
        elif sp.isspmatrix_csc(A):
            self._solve_transposed = True
            self.set_iparm(12, 1)
        else:
            msg = 'PyPardiso requires matrix A to be in CSR or CSC format, but matrix A is: {}'.format(type(A))
            raise TypeError(msg)

        # scipy allows unsorted csr-indices, which lead to completely wrong pardiso results
        if not A.has_sorted_indices:
            A.sort_indices()

        # scipy allows csr matrices with empty rows. a square matrix with an empty row is singular. calling
        # pardiso with a matrix A that contains empty rows leads to a segfault, same applies for csc with
        # empty columns
        if not np.diff(A.indptr).all():
            row_col = 'column' if self._solve_transposed else 'row'
            raise ValueError('Matrix A is singular, because it contains empty {}(s)'.format(row_col))

        if A.dtype != np.float64:
            raise TypeError('PyPardiso currently only supports float64, but matrix A has dtype: {}'.format(A.dtype))

    def _check_b(self, A, b):
        if sp.isspmatrix(b):
            warnings.warn('PyPardiso requires the right-hand side b to be a dense array for maximum efficiency',
                          SparseEfficiencyWarning)
            b = b.todense()

        # pardiso expects fortran (column-major) order if b is a matrix
        if b.ndim == 2:
            b = np.asfortranarray(b)

        if b.shape[0] != A.shape[0]:
            raise ValueError("Dimension mismatch: Matrix A {} and array b {}".format(A.shape, b.shape))

        if b.dtype != np.float64:
            if b.dtype in [np.float16, np.float32, np.int16, np.int32, np.int64]:
                warnings.warn("Array b's data type was converted from {} to float64".format(str(b.dtype)),
                              PyPardisoWarning)
                b = b.astype(np.float64)
            else:
                raise TypeError('Dtype {} for array b is not supported'.format(str(b.dtype)))

        return b

    def _call_pardiso(self, A, b):

        x = np.zeros_like(b)
        pardiso_error = ctypes.c_int32(0)
        c_int32_p = ctypes.POINTER(ctypes.c_int32)
        c_float64_p = ctypes.POINTER(ctypes.c_double)

        # 1-based indexing
        ia = A.indptr + 1
        ja = A.indices + 1

        self._mkl_pardiso(self.pt.ctypes.data_as(ctypes.POINTER(self._pt_type[0])),  # pt
                          ctypes.byref(ctypes.c_int32(1)),  # maxfct
                          ctypes.byref(ctypes.c_int32(1)),  # mnum
                          ctypes.byref(ctypes.c_int32(self.mtype)),  # mtype -> 11 for real-nonsymetric
                          ctypes.byref(ctypes.c_int32(self.phase)),  # phase -> 13
                          ctypes.byref(ctypes.c_int32(A.shape[0])),  # N -> number of equations/size of matrix
                          A.data.ctypes.data_as(c_float64_p),  # A -> non-zero entries in matrix
                          ia.ctypes.data_as(c_int32_p),  # ia -> csr-indptr
                          ja.ctypes.data_as(c_int32_p),  # ja -> csr-indices
                          self.perm.ctypes.data_as(c_int32_p),  # perm -> empty
                          ctypes.byref(ctypes.c_int32(1 if b.ndim == 1 else b.shape[1])),  # nrhs
                          self.iparm.ctypes.data_as(c_int32_p),  # iparm-array
                          ctypes.byref(ctypes.c_int32(self.msglvl)),  # msg-level -> 1: statistical info is printed
                          b.ctypes.data_as(c_float64_p),  # b -> right-hand side vector/matrix
                          x.ctypes.data_as(c_float64_p),  # x -> output
                          ctypes.byref(pardiso_error))  # pardiso error

        if pardiso_error.value != 0:
            raise PyPardisoError(pardiso_error.value)
        else:
            return np.ascontiguousarray(x)  # change memory-layout back from fortran to c order

    def get_iparms(self):
        """Returns a dictionary of iparms"""
        return dict(enumerate(self.iparm, 1))

    def get_iparm(self, i):
        """Returns the i-th iparm (1-based indexing)"""
        return self.iparm[i-1]

    def set_iparm(self, i, value):
        """set the i-th iparm to 'value' (1-based indexing)"""
        if i not in {1, 2, 4, 5, 6, 8, 10, 11, 12, 13, 18, 19, 21, 24, 25, 27, 28, 31, 34, 35, 36, 37, 56, 60}:
            warnings.warn('{} is no input iparm. See the Pardiso documentation.'.format(value), PyPardisoWarning)
        self.iparm[i-1] = value

    def set_matrix_type(self, mtype):
        """Set the matrix type (see Pardiso documentation)"""
        self.mtype = mtype

    def get_max_threads(self):
        """Returns the maximum number of threads the solver will use"""
        return mkl.get_max_threads()

    def set_num_threads(self, num_threads):
        """Set the number of threads the solver should use (only a hint, not guaranteed that
        the solver uses this amount)"""
        mkl.set_num_threads(num_threads)

    def set_statistical_info_on(self):
        """Display statistical info (appears in notebook server console window if pypardiso is
        used from jupyter notebook, use wurlitzer to redirect info to the notebook)"""
        self.msglvl = 1

    def set_statistical_info_off(self):
        """Turns statistical info off"""
        self.msglvl = 0

    def set_phase(self, phase):
        """Set the phase(s) for the solver. See the Pardiso documentation for details."""
        self.phase = phase

    def remove_stored_factorization(self):
        """removes the stored factorization, this will free the memory in python, but the factorization in pardiso
        is still accessible with a direct call to self._call_pardiso(A,b) with phase=33"""
        self.factorized_A = sp.csr_matrix((0, 0))

    def free_memory(self, everything=False):
        """release mkl's internal memory, either only for the factorization (ie the LU-decomposition) or all of
        mkl's internal memory if everything=True"""
        self.remove_stored_factorization()
        A = sp.csr_matrix((0, 0))
        b = np.zeros(0)
        self.set_phase(-1 if everything else 0)
        self._call_pardiso(A, b)
        self.set_phase(13)


class PyPardisoWarning(UserWarning):
    pass


class PyPardisoError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return ('The Pardiso solver failed with error code {}. '
                'See Pardiso documentation for details.'.format(self.value))
