# coding: utf-8
import functools
import scipy.sparse as sp
from .pardiso_wrapper import PyPardisoSolver


# pypardsio_solver is used for the 'spsolve' and 'factorized' functions. Python crashes on windows if multiple
# instances of PyPardisoSolver make calls to the Pardiso library
pypardiso_solver = PyPardisoSolver()


def spsolve(A, b, factorize=True, squeeze=True, solver=pypardiso_solver, *args, **kwargs):
    """
    This function mimics scipy.sparse.linalg.spsolve, but uses the Pardiso solver instead of SuperLU/UMFPACK

        solve Ax=b for x

        --- Parameters ---
        A: sparse square CSR or CSC matrix (scipy.sparse.csr.csr_matrix)
        b: numpy ndarray
           right-hand side(s), b.shape[0] needs to be the same as A.shape[0]
        factorize: boolean, default True
                   matrix A is factorized by default, so the factorization can be reused
        squeeze: default True
                 strange quirk of scipy spsolve, which always returns x.squeeze(), this
                 feature in order to keep it compatible with implementations that rely on
                 this behaviour
        solver: instance of PyPardisoSolver, default pypardiso_solver
                you can supply your own instance of PyPardisoSolver, but using several instances
                of PyPardisoSolver in parallel can lead to errors

        --- Returns ---
        x: numpy ndarray
           solution of the system of linear equations, same shape as b (but returnsshape (n,) if b has shape (n,1))

        --- Notes ---
        The computation time increases only minimally if the factorization and the solve phase are carried out
        in two steps, therefore it is factorized by default. Subsequent calls to spsolve with the same matrix A
        will be drastically faster. This makes the "factorized" method obsolete, but it is kept for compatibility.
    """
    if sp.isspmatrix_csc(A):
        A = A.tocsr()  # fixes issue with brightway2 technosphere matrix

    solver._check_A(A)
    if factorize and not solver._is_already_factorized(A):
        solver.factorize(A)

    x = solver.solve(A, b)

    if squeeze:
        return x.squeeze()  # scipy spsolve always returns vectors with shape (n,) indstead of (n,1)
    else:
        return x


def factorized(A, solver=pypardiso_solver, *args, **kwargs):
    """
    This function mimics scipy.sparse.linalg.factorized, but uses the Pardiso solver instead of SuperLU/UMFPACK

        --- Parameters ---
        A: sparse square CSR or CSC matrix (scipy.sparse.csr.csr_matrix)

        --- Returns ---
        solve_b: callable
                 an array b passed to this callable returns the solution to Ax=b

        --- Notes ---
        The returned callable will store a copy of matrix A. This ensures correct results even when the solver
        was used for other tasks in between calls to the returned callable. The factorization is however not
        stored and the first call will take longer.
        Don't use the factorized method for very large matrices, because it needs to keep an additional copy of
        A in memory.
        !!! Use spsolve directly whenever possible !!! Contrary to the scipy implementation there is no performance
        gain in PyPardiso by using factorized instead of spsolve.

    """
    solve_b = functools.partial(spsolve, A.tocsr().copy(), squeeze=False, solver=solver)

    return solve_b
