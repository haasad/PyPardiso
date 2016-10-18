# coding: utf-8

import pytest
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve as scipyspsolve
from scipy.sparse.linalg import factorized as scipyfactorized

from pypardiso.scipy_aliases import pypardiso_solver, spsolve, factorized
from utils import create_test_A_b_small, create_test_A_b_rand, basic_solve

ps = pypardiso_solver

def test_basic_spsolve_vector():
    ps.remove_stored_factorization()
    ps.free_memory()
    A, b = create_test_A_b_rand()
    xpp = spsolve(A,b)
    xscipy = scipyspsolve(A,b)
    np.testing.assert_array_almost_equal(xpp, xscipy)


def test_basic_spsolve_matrix():
    ps.remove_stored_factorization()
    ps.free_memory()
    A, b = create_test_A_b_rand(matrix=True)
    xpp = spsolve(A,b)
    xscipy = scipyspsolve(A,b)
    np.testing.assert_array_almost_equal(xpp, xscipy)


def test_basic_factorized():
    ps.remove_stored_factorization()
    ps.free_memory()
    A, b = create_test_A_b_rand()
    ppfact = factorized(A)
    xpp = ppfact(b)
    scipyfact = scipyfactorized(A)
    xscipy = scipyfact(b)
    np.testing.assert_array_almost_equal(xpp, xscipy)


def test_factorized_modified_A():
    ps.remove_stored_factorization()
    ps.free_memory()
    assert ps.factorized_A.shape == (0,0)
    A, b = create_test_A_b_small()
    Afact = factorized(A)
    x1 = Afact(b)
    A[4,0] = 27
    x2 = spsolve(A, b)
    assert not np.allclose(x1, x2)
    assert ps.factorized_A[4,0] == 27
    x3 = Afact(b)
    np.testing.assert_array_equal(x1, x3)
    assert ps.phase == 33





