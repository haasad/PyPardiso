
# coding: utf-8

import sys
import ctypes
import warnings
import functools
import mkl
import numpy as np
import scipy.sparse as sp
from scipy.sparse import SparseEfficiencyWarning


__all__ = ['PyPardisoSolver', 'spsolve', 'factorized']


class PyPardisoSolver:
    """
    Python interface to the Intel MKL PARDISO library for solving large sparse linear systems of equations Ax=b.
    
    Pardiso documentation: https://software.intel.com/en-us/node/470282
    
    --- Basic usage ---
    matrix type: real (float64) and nonsymetric
    methods: solve, factorize
    
    - use the "solve(A,b)" method to solve Ax=b for x, where A is a sparse CSR matrix and b is a vector or matrix
    - use the "factorize(A)" method first, if you intend to solve the system more than once for different right-hand 
      sides, the factorization will be reused automatically afterwards
    
    
    --- Advanced usage ---
    methods: get_iparm, get_iparms, set_iparm, set_matrix_type, set_phase
    
    - additional options can be accessed by setting the iparms (see Pardiso documentation for description)
    - other matrix types can be chosen with the "set_matrix_type" method. complex matrix types are currently not
      supported
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
    
    """
    
    def __init__(self, mtype=11, phase=13):
        if sys.platform == 'darwin':
            libmkl_core = ctypes.CDLL('libmkl_core.dylib')
        elif sys.platform == 'win32':
            libmkl_core = ctypes.CDLL('mkl_core.dll')
        else:
            libmkl_core = ctypes.CDLL('libmkl_core.so')
        
        self._mkl_pardiso = libmkl_core.mkl_pds_lp64_pardiso
        
        self.pt = np.zeros(64, dtype=np.int32)
        self.iparm = np.zeros(64, dtype=np.int32)
        self.perm = np.zeros(0, dtype=np.int32)
        
        self.mtype = mtype
        self.phase = phase
        self.msglvl = False
        
        self.factorized_A = None
        
    
    def factorize(self, A):
        """ 
        Factorize the matrix A, the factorization will automatically be used if the same matrix A is passed to the
        solve method. This will drastically increase the speed of solve, if solve is called more than once for the 
        same matrix A
        
        --- Parameters ---
        A: sparse square CSR matrix (scipy.sparse.csr.csr_matrix)
           other matrix types will be converted to CSR
        """
        self.factorized_A = A
        A = self._check_A(A)
        self.set_phase(12)
        b = np.zeros((A.shape[0],1))
        self._call_pardiso(A, b)    
    
    
    def solve(self, A, b):
        """
        solve Ax=b for x
        
        --- Parameters ---
        A: sparse square CSR matrix (scipy.sparse.csr.csr_matrix)
           other matrix types will be converted to CSR
        b: numpy ndarray
           right-hand side(s), b.shape[0] needs to be the same as A.shape[0]
           
        --- Returns ---
        x: numpy ndarray
           solution of the system of linear equations, same shape as b
        """
        if self.factorized_A is A:
            self.set_phase(33)
        else:
            self.set_phase(13)
        
        A = self._check_A(A)
        b = self._check_b(A, b)
        x = self._call_pardiso(A, b)
        
        return x
    
    
    def _check_A(self, A):
        if A.shape[0] != A.shape[1]:
            raise ValueError('Matrix A needs to be square, but has shape: {}'.format(A.shape))
        
        if not sp.isspmatrix_csr(A):
            warnings.warn('pypardiso requires matrix A to be in CSR format for maximum efficiency', 
                          SparseEfficiencyWarning)
            A = sp.csr_matrix(A)
        
        if A.dtype != np.float64:
            raise TypeError('pypardiso currently only works with float64. Matrix A has dtype: {}'.format(A.dtype))
        
        return A
            
            
    def _check_b(self, A, b):
        if sp.isspmatrix(b):
            warnings.warn('pypardiso requires the right-hand side b to be a dense vector/matrix'
                          ' for maximum efficiency', 
                          SparseEfficiencyWarning)
            b = b.todense()
        
        if b.ndim == 1:
            b = b.reshape(len(b), 1)
        
        if b.shape[0] != A.shape[0]:
            raise ValueError("Dimension mismatch: Matrix A {} and vector/matrix b {}".format(A.shape, b.shape))
            
        if b.dtype != np.float64:
            raise TypeError(('pypardiso currently only works with float64. '
                             'Vector/matrix b has dtype: {}'.format(b.dtype)))
        
        return b
        
    
    def _call_pardiso(self, A, b):
        A_data = A.data
        A_ia = A.indptr + 1
        A_ja = A.indices + 1
        x = np.zeros_like(b)
        pardiso_error = ctypes.c_int32(0)
        
        c_int32_p = ctypes.POINTER(ctypes.c_int32)
        c_float64_p = ctypes.POINTER(ctypes.c_double)

        self._mkl_pardiso(self.pt.ctypes.data_as(c_int32_p), # pt
                          ctypes.byref(ctypes.c_int32(1)), # maxfct
                          ctypes.byref(ctypes.c_int32(1)), # mnum
                          ctypes.byref(ctypes.c_int32(self.mtype)), # mtype -> 11 for real-nonsymetric
                          ctypes.byref(ctypes.c_int32(self.phase)), # phase -> 13 
                          ctypes.byref(ctypes.c_int32(A.shape[0])), #N -> number of equations/size of matrix
                          A_data.ctypes.data_as(c_float64_p), # A -> non-zero entries in matrix
                          A_ia.ctypes.data_as(c_int32_p), # ia -> csr-indptr
                          A_ja.ctypes.data_as(c_int32_p), # ja -> csr-indices
                          self.perm.ctypes.data_as(c_int32_p), # perm -> empty
                          ctypes.byref(ctypes.c_int32(b.shape[1])), # nrhs -> number of right-hand sides
                          self.iparm.ctypes.data_as(c_int32_p), # iparm-array
                          ctypes.byref(ctypes.c_int32(self.msglvl)), # msg-level -> 1: statistical info is printed
                          b.ctypes.data_as(c_float64_p), # b -> right-hand side vector/matrix
                          x.ctypes.data_as(c_float64_p), # x -> output
                          ctypes.byref(pardiso_error)) # pardiso error
        
        if pardiso_error.value != 0:
            raise PyPardisoError(pardiso_error.value)
        else:
            return x
        
        
    def get_iparms(self):
        """Returns a dictionary of iparms"""
        return dict(enumerate(self.iparm, 1))
    
    
    def get_iparm(self, i):
        """Returns the i-th iparm (1-based indexing)"""
        return self.iparm[i-1]
    
    
    def set_iparm(self, i, value):
        """set the i-th iparm to 'value' (1-based indexing)"""
        if not i in {1,2,4,5,6,8,10,11,12,13,18,19,21,24,25,27,28,31,34,35,36,37,56,60}:
            warnings.warn('{} is no input iparm. See the Pardiso documentation.'.format(value), UserWarning)
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


# pypardsio_solver is used for the 'spsolve' and 'factorized' functions. Python crashes on windows if multiple 
# instances of PyPardisoSolver make calls to the Pardiso library
pypardiso_solver = PyPardisoSolver()


def spsolve(A, b, factorize=True):
    """
    This function mimics scipy.sparse.linalg.spsolve, but uses the Pardiso solver instead of SuperLU/UMFPACK
    
        solve Ax=b for x
        
        --- Parameters ---
        A: sparse square CSR matrix (scipy.sparse.csr.csr_matrix)
           other matrix types will be converted to CSR
        b: numpy ndarray
           right-hand side(s), b.shape[0] needs to be the same as A.shape[0]
        factorize: boolean, default True
                   matrix A is factorized by default, so the factorization can be reused
           
        --- Returns ---
        x: numpy ndarray
           solution of the system of linear equations, same shape as b
           
        --- Notes ---
        the computation time increases only minimally if the factorization and the solve phase are carried out 
        in two steps, therefore it is factorized by default. Subsequent calls to spsolve with the same matrix A 
        will be drastically faster
    """
    if factorize and not pypardiso_solver.factorized_A is A:
        pypardiso_solver.factorize(A)
    x = pypardiso_solver.solve(A, b)
    return x


def factorized(A):
    """
    This function mimics scipy.sparse.linalg.factorized, but uses the Pardiso solver instead of SuperLU/UMFPACK
    
        --- Parameters ---
        A: sparse square CSR matrix (scipy.sparse.csr.csr_matrix)
           other matrix types will be converted to CSR
           
        --- Returns ---
        solve_b: callable 
        		 a vector/matrix b passed to this callable returns the solution to Ax=b
    """
    pypardiso_solver.factorize(A)
    solve_b = functools.partial(pypardiso_solver.solve, A)
    return solve_b


class PyPardisoError(Exception):
    
    def __init__(self, value):
        self.value = value
    
    def __str__(self):
        return ('The Pardiso solver failed with error code {}. '
                'See Pardiso documentation for details.'.format(self.value))

