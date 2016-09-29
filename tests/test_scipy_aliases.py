# coding: utf-8

import pytest
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve as scipyspsolve
from scipy.sparse.linalg import factorized as scipyfactorized

from pypardiso.scipy_aliases import pypardiso_solver, spsolve, factorized


def create_test_A_b(matrix=False, sort_indices=True):
    """
    --- A ---
    scipy.sparse.csr.csr_matrix, float64
    matrix([[5, 1, 0, 0, 0],
            [0, 6, 2, 0, 0],
            [0, 0, 7, 3, 0],
            [0, 0, 0, 8, 4],
            [0, 0, 0, 0, 9]])
    --- b ---
    np.ndarray, float64
    array([[ 1],
           [ 4],
           [ 7],
           [10],
           [13]])
    or       
    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 7,  8,  9],
           [10, 11, 12],
           [13, 14, 15]])
    
    """
    A = sp.spdiags(np.arange(10, dtype=np.float64).reshape(2,5), [1, 0], 5, 5, format='csr')
    if sort_indices:
        A.sort_indices()
    b = np.arange(1,16, dtype=np.float64).reshape(5,3)
    if matrix:
        return A, b
    else:
        return A, b[:,[0]]

def test_basic_spsolve_vector():
    pypardiso_solver.remove_stored_factorization()
    A, b = create_test_A_b()
    xpp = spsolve(A,b)
    xscipy = scipyspsolve(A,b)
    np.testing.assert_array_almost_equal(xpp, xscipy)


def test_basic_spsolve_matrix():
    pypardiso_solver.remove_stored_factorization()
    A, b = create_test_A_b(matrix=True)
    xpp = spsolve(A,b)
    xscipy = scipyspsolve(A,b)
    np.testing.assert_array_almost_equal(xpp, xscipy)


def test_basic_factorized():
    pypardiso_solver.remove_stored_factorization()
    A, b = create_test_A_b()
    ppfact = factorized(A)
    xpp = ppfact(b)
    scipyfact = scipyfactorized(A)
    xscipy = scipyfact(b)
    np.testing.assert_array_almost_equal(xpp, xscipy)


def test_factorized_modified_A():
    pypardiso_solver.remove_stored_factorization()
    assert pypardiso_solver.factorized_A.shape == (0,0)
    A, b = create_test_A_b()
    Afact = factorized(A)
    assert pypardiso_solver.factorized_A.shape == A.shape
    x1 = Afact(b)
    A[4,0] = 27
    x2 = spsolve(A, b)
    assert not np.allclose(x1, x2)
    assert pypardiso_solver.factorized_A[4,0] == 27
    x3 = Afact(b)
    np.testing.assert_array_equal(x1, x3)
    assert pypardiso_solver.phase == 13 # this is not the desired situation, because now every call to Afact
                                        # will be carried out with phase 13, ie pardiso will factorize everytime
                                        # -> needs to be documented in list of caveats or similar






