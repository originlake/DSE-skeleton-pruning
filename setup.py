from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        'dsepruning.dse_helper',
        ['dsepruning/dse_helper.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']
    )
]

setup(
    ext_modules=cythonize(extensions)
)
