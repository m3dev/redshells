from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = [
    'luigi',
    'gokart>=0.1.20',
    'python-dateutil==2.7.5',
    'pandas',
    'scipy',
    'numpy',
    'gensim',
    'scikit-learn',
    'tensorflow>=1.13.1, <2.0',
    'tqdm',
    'optuna==0.6.0',
    'docutils==0.15' # to avoid dependency conflict
]

setup(
    name='redshells',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    description='Tasks which are defined using gokart.TaskOnKart. The tasks can be used with data pipeline library "luigi".',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='M3, inc.',
    url='https://github.com/m3dev/redshells',
    license='MIT License',
    packages=find_packages(),
    install_requires=install_requires,
    test_suite='test')
