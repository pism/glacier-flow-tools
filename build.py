from setuptools import Extension
from setuptools.command.build_py import build_py as _build_py
import numpy

# If the user set NO_OPENMP, proceed with these options. Otherwise add options clang uses.
libraries=['gsl', 'gslcblas']
extra_compile_args=["-O3", "-ffast-math", "-Wall"]
extra_link_args=[]
try:
    os.environ["NO_OPENMP"]
except:
    extra_compile_args.append('-fopenmp')
    libraries.append('gomp')

class build_py(_build_py):
    def run(self):
        self.run_command("build_ext")
        return super().run()

    def initialize_options(self):
        super().initialize_options()
        if self.distribution.ext_modules == None:
            self.distribution.ext_modules = []

        self.distribution.ext_modules.append(
            Extension("pism_dbg",
                      sources=["python/pism_dbg.pyx",
                               "src/upslope_area.cc",
                               "src/accumulated_flow.cc",
                               "src/initialize_mask.cc",
                               "src/DEM.cc"
                               ],
                      include_dirs=[numpy.get_include()],
                      libraries=libraries,
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args,
                      language="c++"
                      )
        )
