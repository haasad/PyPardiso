# coding: utf-8

from io import open
from setuptools import setup

setup(
    name='pypardiso',
    version="0.1.0",
    packages=['pypardiso'],
    author="Adrian Haas",
    author_email="adrian.pypardiso@gmail.com",
    license=open('LICENSE.txt', encoding='utf8').read(),
    #install_requires=['numpy', 'scipy'],  # this breaks conda build, and is probably not necessary anyway
    url="https://github.com/haasad/PyPardisoProject",
    long_description=open('README.md', encoding='utf8').read(),
    description='Python interface to the Intel MKL Pardiso library to solve large sparse linear systems of equations',
    #classifiers=[
    #    'Intended Audience :: End Users/Desktop',
    #    'Intended Audience :: Developers',
    #    'Intended Audience :: Science/Research',
    #    'License :: OSI Approved :: BSD License',
    #    'Operating System :: MacOS :: MacOS X',
    #    'Operating System :: Microsoft :: Windows',
    #    'Operating System :: POSIX',
    #    'Programming Language :: Python',
    #    'Programming Language :: Python :: 3',
    #    'Programming Language :: Python :: 3.4',
    #    'Programming Language :: Python :: 3.5',
    #    'Topic :: Scientific/Engineering :: Information Analysis',
    #    'Topic :: Scientific/Engineering :: Mathematics',
    #    'Topic :: Scientific/Engineering :: Visualization',
    #],
)