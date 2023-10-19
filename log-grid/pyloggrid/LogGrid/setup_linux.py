"""Cython settings for linux"""
import os
from distutils.core import setup
from distutils.extension import Extension

import numpy as np
from Cython.Build import cythonize

os.environ["CC"] = "clang"
os.environ["CXX"] = "clang -shared"

ext_modules = [
    Extension(
        "compute_convolution_kernel",
        ["compute_convolution_kernel.pyx"],
        extra_compile_args=["-fopenmp", "-march=native", "-c", "-g", "-Wextra", "-march=native", "-ffast-math", "-funroll-loops", "-fno-stack-protector"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    ext_modules=cythonize(ext_modules, annotate=True),
    include_dirs=[np.get_include()],
    name="log-grid",
    author="A. Barral",
)
