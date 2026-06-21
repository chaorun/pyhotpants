# ====== DEPRECATED ======
# Cython 编译已弃用，所有功能已迁移至纯 Python numutils.py
# 安装方式：pip install -e . 或直接 PYTHONPATH=. python
# ====== DEPRECATED ======
#
# from setuptools import setup, Extension
# from Cython.Build import cythonize
# import numpy as np
#
# ext = Extension(
#     "chotpants",
#     sources=[
#         "chotpants.pyx",
#         "csrc/hotpants_compute.c",
#         "csrc/alard.c",
#         "csrc/functions.c",
#     ],
#     include_dirs=[
#         "csrc",
#         np.get_include(),
#         "/opt/homebrew/include",
#     ],
#     library_dirs=["/opt/homebrew/lib"],
#     libraries=["m", "cfitsio"],
#     extra_compile_args=["-funroll-loops", "-O3", "-ansi", "-std=c99"],
# )
#
# setup(
#     name="pyhotpants",
#     ext_modules=cythonize([ext], compiler_directives={"language_level": "3"}),
# )
