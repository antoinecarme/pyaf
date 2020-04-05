from setuptools import setup
from setuptools import find_packages

setup(name='pyaf',
      version='1.2.3',
      description='Python Automatic Forecasting',
      author='Antoine CARME',
      author_email='antoine.carme@laposte.net',
      url='https://github.com/antoinecarme/pyaf',
      license='BSD 3-clause',
      packages=find_packages(include=['pyaf.*']),
      python_requires='>=3',
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Programming Language :: Python :: 3'],
      keywords='arx automatic-forecasting autoregressive benchmark cycle decomposition exogenous forecasting heroku hierarchical-forecasting horizon jupyter pandas python scikit-learn seasonal time-series transformation trend web-service',
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
