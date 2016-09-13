# coding: utf-8
import functools
from .pardiso_wrapper import PyPardisoSolver


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

