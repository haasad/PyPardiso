# coding: utf-8

import pytest
import numpy as np
import scipy.sparse as sp
from utils import create_test_A_b_small, create_test_A_b_rand, basic_solve
from pypardiso.scipy_aliases import pypardiso_solver

ps = pypardiso_solver


def test_input_A_unsorted_indices():
    A, b = create_test_A_b_small(sort_indices=False)
    ps._check_A(A)
    assert A.has_sorted_indices
    basic_solve(A, b)


def test_input_A_non_sparse():
    A, b = create_test_A_b_rand(10, 0.9)
    A = A.todense()
    assert not sp.issparse(A)
    with pytest.raises(TypeError):
        ps.solve(A, b)


def test_input_A_csc():
    A, b = create_test_A_b_rand()
    x_csr = ps.solve(A, b)
    Acsc = A.copy().asformat('csc')
    x_csc = ps.solve(Acsc, b)
    np.testing.assert_array_almost_equal(x_csr, x_csc)


def test_input_A_other_sparse():
    A, b = create_test_A_b_rand(50)
    for f in ['coo', 'lil', 'dia']:
        Aother = A.asformat(f)
        with pytest.raises(TypeError):
            ps.solve(Aother, b)


def test_input_A_empty_row_and_col():
    A, b = create_test_A_b_rand(25, 0.7)
    A = np.array(A.todense())
    A[0, :] = 0
    A[:, 0] = 0
    Asp = sp.csr_matrix(A)
    with pytest.raises(ValueError):
        ps.solve(Asp, b)


def test_input_A_empty_row():
    A, b = create_test_A_b_rand(25, 0.7)
    A = np.array(A.todense())
    A[0, :] = 0
    A[1, 0] = 1
    Asp = sp.csr_matrix(A)
    with pytest.raises(ValueError):
        basic_solve(Asp, b)


def test_input_A_dtypes():
    A, b = create_test_A_b_rand(10, 0.5)
    for d in [np.float16, np.float32, np.int16, np.int32, np.int64, np.complex64,
              np.complex128, np.complex128, np.uint16, np.uint32, np.uint64]:
        with pytest.raises(TypeError):
            ps.solve(A.astype(d), b)


# def test_input_A_empty_col():
#    A, b = create_test_A_b_rand(25, 0.1)
#    A = np.array(A.todense())
#    A[0,:] = 0
#    A[:,0] = 0
#    A[0, 1] = 1
#    Asp = sp.csr_matrix(A)
#    with pytest.warns(PyPardisoWarning):
#        x = ps.solve(A,b)
#        print(x)

# dosen't necessarily return inf value, can also be just really big


def test_input_A_nonsquare():
    A, b = create_test_A_b_rand()
    A = sp.csr_matrix(np.concatenate([A.todense(), np.ones((A.shape[0], 1))], axis=1))
    with pytest.raises(ValueError):
        basic_solve(A, b)
