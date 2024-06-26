"""
Installs Cython bindings.
"""

from Cython.Build import cythonize
from setuptools import Extension, setup

ext_modules = [
    Extension(
        "dem_smoothing",
        ["dem_smoothing.pyx"],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name="dem_smoothing",
    ext_modules=cythonize(ext_modules),
)
