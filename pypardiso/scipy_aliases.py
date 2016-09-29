# coding: utf-8
import functools
from .pardiso_wrapper import PyPardisoSolver


# pypardsio_solver is used for the 'spsolve' and 'factorized' functions. Python crashes on windows if multiple 
# instances of PyPardisoSolver make calls to the Pardiso library
pypardiso_solver = PyPardisoSolver()


def spsolve(A, b, *args, factorize=True, solver=pypardiso_solver, **kwargs):
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
    A = solver._check_A(A)
    if factorize and not solver._is_already_factorized(A):
        pypardiso_solver.factorize(A)
        
    x = pypardiso_solver.solve(A, b)
    
    return x.squeeze() # scipy always returns vectors with shape (n,) indstead of (1,n)


def factorized(A, *args, solver=pypardiso_solver, **kwargs):
    """
    This function mimics scipy.sparse.linalg.factorized, but uses the Pardiso solver instead of SuperLU/UMFPACK
    
        --- Parameters ---
        A: sparse square CSR matrix (scipy.sparse.csr.csr_matrix)
           other matrix types will be converted to CSR
           
        --- Returns ---
        solve_b: callable 
                 a vector/matrix b passed to this callable returns the solution to Ax=b
        
        --- Notes ---
        The returned callable will store a copy of matrix A. This ensures correct results even when the solver
        was used for other tasks in between calls to the returned callable. The factorization is however not 
        stored and the first call will take longer.
                 
    """
    A = solver._check_A(A)
    solver.factorize(A)
    solve_b = functools.partial(solver.solve, A.copy())
    
    return solve_b

