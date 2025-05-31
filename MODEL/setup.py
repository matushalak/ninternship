from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

# Define the extension explicitly
extensions = [
    Extension(
        name="MODEL.adex_euler",
        sources=[os.path.join("MODEL", "adex_euler.pyx")],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name="nin",
    ext_modules=cythonize(extensions),
    zip_safe=False
)