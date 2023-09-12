from setuptools import setup, Extension
import numpy

setup(
    ext_modules=[
        Extension(
            'dsepruning.dse_helper',
            ['dsepruning/dse_helper.pyx'],
            include_dirs=[numpy.get_include()],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp']
        )
    ]
)
