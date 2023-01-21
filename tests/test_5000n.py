# coding: utf-8
from utils import create_test_A_b_rand, basic_solve
from pypardiso.scipy_aliases import pypardiso_solver

ps = pypardiso_solver


def test_larger_system():
    A, b = create_test_A_b_rand(n=5000)
    basic_solve(A, b)
