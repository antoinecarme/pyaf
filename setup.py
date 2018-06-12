from setuptools import setup
from setuptools import find_packages

from shutil import copyfile, rmtree
import os
import glob
setup(name='pyaf',
      version='1.0',
      description='Python Automatic Forecasting',
      author='Antoine CARME',
      author_email='antoine.carme@laposte.net',
      url='https://github.com/antoinecarme/pyaf',
      license='BSD 3-clause',
      packages=['pyaf'],
      install_requires=[
          'scipy',
          'pandas',
          'sklearn',
          'matplotlib',
          'pydot',
          'dill',
          'pathos',
          'sqlalchemy'
      ])

