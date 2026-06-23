import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

lib_dirs = os.environ.get("CFITSIO_LIB", "/usr/local/lib").split(":")
inc_dirs = os.environ.get("CFITSIO_INC", "/usr/local/include").split(":")

ext = Extension(
    "chotpants",
    sources=[
        "chotpants.pyx",
        "csrc/hotpants_compute.c",
        "csrc/alard.c",
        "csrc/functions.c",
    ],
    include_dirs=["csrc", np.get_include()] + inc_dirs,
    library_dirs=lib_dirs,
    libraries=["m", "cfitsio"],
    extra_compile_args=["-funroll-loops", "-O3", "-ansi", "-std=c99"],
)

setup(
    name="pyhotpants-cython",
    ext_modules=cythonize([ext], compiler_directives={"language_level": "3"}),
)
