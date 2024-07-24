from setuptools import find_packages, setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    name='src',
    packages=find_packages()
    )