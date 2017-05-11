from setuptools import setup
from setuptools import find_packages

from shutil import copyfile, rmtree
import os
import glob

def build_package():
    try:
        rmtree('pyaf')
    except:
        pass
    os.mkdir('pyaf')
    os.mkdir('pyaf/TS')
    for file in glob.glob('TS/*.py'):        
        copyfile(file, 'pyaf/' + file)
    copyfile('ForecastEngine.py' ,
             'pyaf/ForecastEngine.py')
    copyfile('HierarchicalForecastEngine.py',
             'pyaf/HierarchicalForecastEngine.py')
    os.mkdir('pyaf/CodeGen')
    for file in glob.glob('CodeGen/*.py'):        
        copyfile(file, 'pyaf/' + file)
    os.mkdir('pyaf/Bench')
    for file in glob.glob('Bench/*.py'):        
        copyfile(file, 'pyaf/' + file)
    

build_package();

setup(name='pyaf',
      version='1.0',
      description='Python Automatic Forecasting',
      author='Antoine CARME',
      author_email='antoine.carme@laposte.net',
      url='https://github.com/antoinecarme/pyaf',
      license='BSD 3-clause',
      packages=['pyaf' , 'pyaf.TS' ,  'pyaf.CodeGen' ,  'pyaf.Bench'],
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

rmtree('pyaf')
