import sys, os, subprocess
from setuptools import setup, find_packages

gpu = None
use_cython = False

args = []
for arg in sys.argv:
    if arg.startswith("--gpu="):
        gpu = arg.split("=", 1)[1]
    elif arg == "--cython":
        use_cython = True
    else:
        args.append(arg)

sys.argv = args

install_requires = ["numpy", "numba", "scipy"]

if gpu == "metal":
    install_requires.append("mlx")
elif gpu and gpu not in ("cuda64", "agx_orin"):
    print(f"WARNING: unknown --gpu={gpu}, ignored. Valid: metal, cuda64, agx_orin")

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pyhotpants",
    packages=find_packages(exclude=["deprecated", "*.deprecated", "deprecated.*"]),
    python_requires=">=3.8",
    install_requires=install_requires,
    description="hotpants 天文图像差分算法 — Python 实现",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
    ],
)

if use_cython:
    print("\n=== Building Cython backend ===\n")
    subprocess.check_call([sys.executable, "setup.py", "install"], cwd="cython")
