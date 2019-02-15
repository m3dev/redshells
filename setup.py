from setuptools import setup, find_packages

readme_note = """\
.. note::

   For the latest source, discussion, etc, please visit the
   `GitHub repository <https://github.com/m3dev/redshells>`_\n\n
"""

with open('README.md') as f:
    long_description = readme_note + f.read()

install_requires = [
    'luigi',
    'gokart',
    'pandas',
    'numpy',
    'gensim',
    'scikit-learn',
    'tensorflow',
    'tqdm',
    'optuna==0.6.0'
]

setup(
    name='redshells',
    version='0.1.2',
    description=
    'Tasks which are defined using gokart.TaskOnKart. The tasks can be used with data pipeline library "luigi".',
    long_description=long_description,
    author='M3, inc.',
    url='https://github.com/m3dev/redshells',
    license='MIT License',
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        'Programming Language :: Python :: 3.7',
    ],
)
