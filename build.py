"""
This module is used to build a Python extension module named "pism_dbg" using setuptools.

The module uses the GSL (GNU Scientific Library) and optionally OpenMP for parallel processing.
The source code for the extension module is written in C++ and includes several source files.

initialize_options()
    Initializes options for the build process. If no extension modules are defined, it initializes an empty list and adds the "pism_dbg" extension to it.

run()
    Runs the build_ext command before running the original build_py command.
"""

import os

import numpy
from setuptools import Extension
from setuptools.command.build_py import build_py as _build_py

# If the user set NO_OPENMP, proceed with these options. Otherwise add options clang uses.
libraries = ["gsl", "gslcblas"]
extra_compile_args = ["-O3", "-ffast-math", "-Wall"]

try:
    os.environ["NO_OPENMP"]
except:
    extra_compile_args.append("-fopenmp")
    libraries.append("gomp")


class build_py(_build_py):
    """
    Class used to build extentions.
    """

    def run(self):
        """
        Execute the build_ext command before running the original build_py command.

        Returns
        -------
        Any
            The result of the superclass's run method.
        """
        self.run_command("build_ext")
        return super().run()

    def initialize_options(self):
        """
        Initialize options.
        """
        super().initialize_options()
        if self.distribution.ext_modules is None:
            self.distribution.ext_modules = []

        self.distribution.ext_modules.append(
            Extension(
                "pism_dbg",
                sources=[
                    "python/pism_dbg.pyx",
                    "src/upslope_area.cc",
                    "src/accumulated_flow.cc",
                    "src/initialize_mask.cc",
                    "src/DEM.cc",
                ],
                include_dirs=[numpy.get_include()],
                libraries=libraries,
                extra_compile_args=extra_compile_args,
                language="c++",
            )
        )
