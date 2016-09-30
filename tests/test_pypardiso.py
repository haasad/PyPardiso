# coding: utf-8

import pytest
import numpy as np
import scipy.sparse as sp
from scipy.sparse import SparseEfficiencyWarning
from scipy.sparse.linalg import spsolve as scipyspsolve
from scipy.sparse.linalg import factorized as scipyfactorized

from pypardiso.pardiso_wrapper import PyPardisoSolver, PyPardisoError, PyPardisoWarning
from pypardiso.scipy_aliases import pypardiso_solver, spsolve, factorized

ps = pypardiso_solver


#def create_test_A_b(matrix=False, sort_indices=True):
#    """
#    --- A ---
#    scipy.sparse.csr.csr_matrix, float64
#    matrix([[5, 1, 0, 0, 0],
#            [0, 6, 2, 0, 0],
#            [0, 0, 7, 3, 0],
#            [0, 0, 0, 8, 4],
#            [0, 0, 0, 0, 9]])
#    --- b ---
#    np.ndarray, float64
#    array([[ 1],
#           [ 4],
#           [ 7],
#           [10],
#           [13]])
#    or       
#    array([[ 1,  2,  3],
#           [ 4,  5,  6],
#           [ 7,  8,  9],
#           [10, 11, 12],
#           [13, 14, 15]])
#    
#    """
#    A = sp.spdiags(np.arange(10, dtype=np.float64).reshape(2,5), [1, 0], 5, 5, format='csr')
#    if sort_indices:
#        A.sort_indices()
#    b = np.arange(1,16, dtype=np.float64).reshape(5,3)
#    if matrix:
#        return A, b
#    else:
#        return A, b[:,[0]]

def create_test_A_b(n=1000, density=0.5, matrix=False, sort_indices=True):
    A = sp.csr_matrix(sp.rand(n, n, density) + sp.eye(n))
    if matrix:
        b = np.random.rand(n,1)
    else:
        b = np.random.rand(n,5)
    
    return A, b


def basic_solve(A,b):
    x = ps.solve(A,b)
    np.testing.assert_array_almost_equal(A*x, b)


def test_bvector_smoketest():
    A, b = create_test_A_b()
    basic_solve(A,b)


def test_bmatrix_smoketest():
    A, b = create_test_A_b(matrix=True)
    basic_solve(A,b)


def test_input_A_unsorted_indices():
    A, b = create_test_A_b(sort_indices=False)
    assert not A.has_sorted_indices
    assert ps._check_A(A).has_sorted_indices
    basic_solve(A,b)


def test_input_A_non_sparse():
    A, b = create_test_A_b()
    A = A.todense()
    assert not sp.issparse(A)
    with pytest.warns(SparseEfficiencyWarning):
        basic_solve(A,b)


def test_input_A_other_sparse():
    A, b = create_test_A_b()
    for f in ['bsr', 'coo', 'csc', 'dia', 'dok', 'lil']:
        Aother = A.asformat(f)
        with pytest.warns(SparseEfficiencyWarning):
            basic_solve(Aother, b)


def test_input_A_empty_row_and_col():
    A, b = create_test_A_b()
    A = A.tolil()
    A[0,:] = 0
    A = A.tocsr()
    with pytest.raises(ValueError):
        basic_solve(A, b)


def test_input_A_empty_row():
    A, b = create_test_A_b()
    A = A.tolil()
    A[0,:] = 0
    A[1, 0] = 1
    A = A.tocsr()
    with pytest.raises(ValueError):
        basic_solve(A,b)


def test_input_A_empty_col():
    A, b = create_test_A_b()
    A = A.tolil()
    A[0,:] = 0
    A[0, 1] = 1
    A = A.tocsr()
    with pytest.warns(PyPardisoWarning):
        x = ps.solve(A,b)


def test_input_A_dtypes():
    A, b = create_test_A_b()
    for dt in [np.float16, np.float32, np.int16, np.int32, np.int64]:
        Adt = A.astype(dt)
        with pytest.warns(PyPardisoWarning):
            basic_solve(Adt, b)
            
    for dt in [np.complex64, np.complex128, np.complex128, np.uint16, np.uint32, np.uint64]:
        Adt = A.astype(dt)
        with pytest.raises(TypeError):
            basic_solve(Adt, b)


def test_input_A_nonsquare():
    A, b = create_test_A_b()
    A = sp.csr_matrix(np.concatenate([A.todense(), np.ones((A.shape[0], 1))], axis=1))
    with pytest.raises(ValueError):
        basic_solve(A,b)


def test_input_b_sparse():
    A, b = create_test_A_b()
    for sparse_format in [sp.csr_matrix, sp.csc_matrix, sp.lil_matrix, sp.coo_matrix]:
        bsparse = sparse_format(b)
        with pytest.warns(SparseEfficiencyWarning):
            x = ps.solve(A, bsparse)
            np.testing.assert_array_almost_equal(A*x, b)


def test_input_b_shape():
    A, b = create_test_A_b()
    x_array = ps.solve(A,b)
    assert x_array.shape == b.shape
    x_vector = ps.solve(A, b.ravel())
    assert x_vector.shape == b.ravel().shape
    np.testing.assert_array_equal(x_array.ravel(), x_vector)


def test_input_b_dtypes():
    A, b = create_test_A_b()
    for dt in [np.float16, np.float32, np.int16, np.int32, np.int64]:
        bdt = b.astype(dt)
        with pytest.warns(PyPardisoWarning):
            basic_solve(A, bdt)
            
    for dt in [np.complex64, np.complex128, np.complex128, np.uint16, np.uint32, np.uint64]:
        bdt = b.astype(dt)
        with pytest.raises(TypeError):
            basic_solve(A, bdt)


def test_input_b_fortran_order():
    A, b = create_test_A_b(matrix=True)
    x = ps.solve(A,b)
    xfort = ps.solve(A, np.asfortranarray(b))
    np.testing.assert_array_equal(x, xfort)


def test_input_b_wrong_shape():
    A, b = create_test_A_b()
    b = np.append(b, 1)
    with pytest.raises(ValueError):
        basic_solve(A,b)


def test_factorization_is_used():
    ps.remove_stored_factorization()
    A, b = create_test_A_b()
    x1 = ps.solve(A,b)
    assert ps.phase == 13
    assert ps.factorized_A.shape == (0,0)
    ps.factorize(A)
    x2 = ps.solve(A,b)
    assert ps.phase == 33
    x3 = ps._call_pardiso(sp.csr_matrix(A.shape), b)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(x2, x3)


def test_no_stale_factorization_used_array_equal():
    ps.remove_stored_factorization()
    A, b = create_test_A_b()
    ps.factorize(A)
    x1 = ps.solve(A,b)
    assert ps.phase == 33
    A[4,0] = 27
    x2 = ps.solve(A,b)
    assert ps.phase == 13
    assert not np.allclose(x1, x2)


def test_csr_hashing():
    ps.remove_stored_factorization()
    A, b = create_test_A_b()
    assert sp.isspmatrix(ps.factorized_A)
    ps.size_limit_storage = 0
    ps.factorize(A)
    assert type(ps.factorized_A) == str
    x = ps.solve(A,b)
    assert ps.phase == 33
    ps.size_limit_storage = 5e7


def test_no_stale_factorization_used_hashing():
    ps.remove_stored_factorization()
    A, b = create_test_A_b()
    ps.size_limit_storage = 0
    ps.factorize(A)
    assert type(ps.factorized_A) == str
    x1 = ps.solve(A,b)
    assert ps.phase == 33
    A[4,0] = 27
    x2 = ps.solve(A,b)
    assert ps.phase == 13
    assert not np.allclose(x1, x2)
    ps.size_limit_storage = 5e7


# scipy aliases

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