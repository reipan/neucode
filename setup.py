# Package metadata lives in pyproject.toml ([project] table).
# This file exists only to build the Cython simulation-core extension, which
# cannot yet be expressed declaratively in pyproject.toml.
import glob

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extension = Extension(
    name="neucode.simcore",
    sources=[
        "neucode/simcore.pyx",
        *glob.glob("neucode/c_src/src/*.c"),
    ],
    include_dirs=[
        "neucode/c_src/include",
        numpy.get_include(),
    ],
    language="c",
)

setup(ext_modules=cythonize([extension]))
