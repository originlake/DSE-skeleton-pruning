from setuptools import setup, find_packages, Extension

class get_numpy_include(object):

    def __str__(self):
        import numpy
        return numpy.get_include()

setup(
    name="dsepruning",
    version = '0.1.0',
    packages=find_packages(include = ['dsepruning', 'dsepruning.*']),
    description="Python implementation of discrete skeleton evolution",
    url="https://https://github.com/originlake/DSE-skeleton-pruning",
    setup_requires=[
        "cython",
        "numba",
        "numpy<1.24,>=1.18"
    ],
    install_requires=[
        "numba",
        "sknw",
        "scikit-image",
        "networkx>2.3",
    ],
    ext_modules=[
        Extension(
            'dsepruning.dse_helper',
            ['dsepruning/dse_helper.pyx'],
            include_dirs=[get_numpy_include()],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp']
        )
    ]
)