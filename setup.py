from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    pyaf_long_description = fh.read()
    
setup(name='pyaf',
      version='5.0',
      description='Python Automatic Forecasting',
      long_description=pyaf_long_description,
      long_description_content_type="text/markdown",
      author='Antoine CARME',
      author_email='antoine.carme@outlook.com',
      url='https://github.com/antoinecarme/pyaf',
      license='BSD 3-clause',
      packages=find_packages(include=['pyaf', 'pyaf.*']),
      python_requires='>=3',
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Programming Language :: Python :: 3'],
      keywords='arx automatic-forecasting autoregressive benchmark cycle decomposition exogenous forecasting heroku hierarchical-forecasting horizon jupyter pandas python scikit-learn seasonal time-series transformation trend web-service',
      install_requires=[
          'scipy',
          'pandas',
          'scikit-learn',
          'matplotlib',
          'pydot',
          'dill',
      ])
