from setuptools import setup

descr = """dsepruning: Discrete skeleton evolution, a keleton pruning algorithm 
"""

if __name__ == '__main__':
    setup(name='dsepruning',
        version='0.1',
        url='https://github.com/originlake/DSEpruning',
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
