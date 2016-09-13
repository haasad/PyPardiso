# coding: utf-8

from .pardiso_wrapper import PyPardisoSolver
from .scipy_aliases import spsolve, factorized

__version__='0.1.0'
__all__ = ['PyPardisoSolver', 'spsolve', 'factorized']