# coding: utf-8
from importlib.metadata import version, PackageNotFoundError

from .pardiso_wrapper import PyPardisoSolver
from .scipy_aliases import spsolve, factorized
from .scipy_aliases import pypardiso_solver as ps


try:
    __version__ = version(__package__)
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ['PyPardisoSolver', 'spsolve', 'factorized', 'ps']
