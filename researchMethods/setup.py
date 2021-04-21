from setuptools import find_packages
from setuptools import setup

# setup.py based on a src/ package structure
# https://blog.ionelmc.ro/2014/05/25/python-packaging

setup(
    name='fffit',
    version='0.1',
    description='A meta-heuristic package to fit molecular dynamics forcefields to data.',
    author='Elton Carvalho, Jhonat Sousa',
    author_email='elton.carvalho@ect.ufrn.br',
    # url='https://github.com/eltonfc/fffit',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    # py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    #project_urls={
    #    'Changelog': 'https://github.com/ionelmc/python-nameless/blob/master/CHANGELOG.rst',
    #    'Issue Tracker': 'https://github.com/ionelmc/python-nameless/issues',
    #},
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
        'PSO',
        'particle swarm optimization',
        'molecular dynamics',
        'force field',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy'
    ],
    extras_require={
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
    setup_requires=[
        # 'pytest-runner',
    ],
    entry_points={
        # 'console_scripts': [
        #    'nameless = nameless.cli:main',
        # ]
    },
)

