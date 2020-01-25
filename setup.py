from setuptools import setup

descr = """dse: Discrete skeleton evolution, a keleton pruning algorithm 
"""

if __name__ == '__main__':
    setup(name='dse',
        version='0.1',
        url='https://github.com/originlake/DSE',
        description='skeleton pruning method',
        long_description=descr,
        author='Shuo Z',
        author_email='shuozhong0331@gmail.com',
        license='MIT License',
        packages=['dse'],
        package_data={},
        install_requires=[
            'numpy',
            'sknw',
            'networkx==2.3',
            'numba',
            'scikit-image'
            ],
    )
