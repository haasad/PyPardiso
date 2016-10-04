# coding: utf-8

from utils import create_test_A_b_small, create_test_A_b_rand, basic_solve
from pypardiso.scipy_aliases import pypardiso_solver

ps = pypardiso_solver


def test_bvector_smoketest():
	A, b = create_test_A_b_rand()
	#print(ps.get_iparms())
	basic_solve(A,b)
	ps.free_memory()


def test_bmatrix_smoketest():
    A, b = create_test_A_b_rand(matrix=True)
    basic_solve(A,b)
    ps.free_memory()