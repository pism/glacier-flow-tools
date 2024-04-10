import os
import sys
from distutils.core import setup

PKG_NAME = "pypism"

setup(
    name=PKG_NAME,
    version="0.0.99",
    description="Python Tools",
    author="Andy Aschwanden",
    author_email="aaschwanden@alaska.edu",
    url="https://github.com/pism/pypismtools",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Topic :: Utilities",
    ],
    packages=[PKG_NAME],
    package_dir={PKG_NAME: "."},
)
