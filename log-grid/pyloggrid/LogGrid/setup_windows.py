"""Cython settings for windows"""
from distutils.core import setup

import numpy
from Cython.Build import cythonize
from setuptools import Extension

ext_modules = [
    Extension(
        "compute_convolution_kernel",
        ["compute_convolution_kernel.pyx"],
        extra_compile_args=["/openmp"],
    )
]
setup(
    ext_modules=cythonize(ext_modules, annotate=True, compiler_directives={"language_level": "3"}),
    include_dirs=[numpy.get_include()],
    name="log-grid",
    author="A. Barral",
)
