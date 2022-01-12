import numpy as np
import scipy.sparse as sp

from utils import create_test_A_b_rand, basic_solve
from pypardiso.scipy_aliases import pypardiso_solver

def test_readme_example():
    A = sp.rand(10, 10, density=0.5, format='csr')
    b = np.random.rand(10)
    basic_solve(A, b)

def test_issue36():
    A = sp.rand(5000, 5000, density=0.01, format='csr')
    b = np.random.rand(5000)
    basic_solve(A, b)
