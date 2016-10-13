# Author: Antoine Carme <antoine.carme@laposte.net>
# License: BSD 3 clause

PyAF (Python Automatic Forecasting)
===================================

PyAF is a Python module for Automatic Forecasting built on top of popular data science python modules : numpy, scipy, pandas and scikit-learn. 

PyAF works as an automated way of predicting future values of a signal using a machien learning approach. It provides a set of features similar to some commercial automatic forecasting products.

PyAF is distributed under the 3-Clause BSD license.

PyAF is currently maintained by the original developer.

Features
--------
PyAF allows forecasting a time series for future values in a fully automated way. 

It uses pandas as a data access layer. It consumes data from a pandas dataframe (with time and signal columns), builds a time series model, and outputs the forecasts in a pandas dataframe. Pandas is an excellent data access layer, it allows reading/writing a huge set of formats, accessing differnet datasources (databases) and has an extensive set of algorithms to handle dataframes (aggregation, statstics etc).

Statistical time series models are built using scikit-learn.

As of Oct. 12 2016, the following features were available :
1. Training a model to forecast a time series (given in a pandas dataframe with time and signal columns).
	1.1 it used a machine learning approach (the signal is cut into Estimation and validation parts, respectively, 80% and 20% of the signal).
2. Forecasting a time series model on a given horizon (forecast result is also pandabs dataframe).  
3. Training features
  3.1 Signal decomposition as the sum of a trend, periodic and AR component (https://en.wikipedia.org/wiki/Decomposition_of_time_series)
  3.2 It works as a competition between a huge set of possible decopomsitions. a set of possible trends, periodic components and AR models is generated and all the possible comibinations are estimated. The best decomposition in term of performance is kept to forecast the signal (the performance is computed on a part od the signal that was not used for the estimation).
  3.3 All Models are estimated using standard procedures and state-pof-the-art time series modeling. for example, trend regressions and AR models are estimated using scikit learn ridge regressions. 
  3.4 Standard performance measures are used (L1, RMSE, MAPE, etc)
  3.5 . Allows prediction intervals for the forecasts.
4. Exogenous data can be provided to improve the forecasts. These are expected to be stored in an external dataframe.
  4.1 the user prodvieds exogenous data in an external dataframe (this dataframe is merged with the training dataframe)
  4.2 Exogenous data are integrated in the mdoeling process through thoir past values (ARX models, https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model).
  4.3 Exogenous variables can be of any type (numeric or string , date, object). 
  4.4 Exogenous variables are dummified for the non-numeric case, and standardized for the numeric types. 
5. A hierarchical forecasting is starting. It follows the excellent Rob J Hyndman and George Athanasopoulos book approach (https://www.otexts.org/fpp/9/4).
	5.1 hierarchies and grouped time series are supported. 
	5.2 boottom-up, top-down (using proportions) and optimal combinations are supported. 

4. It is customizable and has a huge set of options. The default set of options should however be OK.
5. basic lotting fiunctions using matplotlib.
5. An object-oriented approach is used for the system design. Separation of concerns is the key factor here.
6. Fully written in python with numpy, scipy, pandas and scikit-learn objects. Tries to be columnar-based  for perforamnce reasons.  
6. A test-driven approach is used. A lot of exampels are available in the test directory.
7. A benchmarking process is expected (M1, M2, M3 compeiuttions, NN3, NN5, etc). 
8. Some jupyter notebooks are available for demo purposes.
9. A basic web services (Flask) effort is starting.
10. A project for SQL generation is started (using sqlalchemy). the goal is to be able to export the forecasts as a SQL code. This is a WIP.

PyAF is a work in progress. The set of features is evloving. Your comments, help, hints are welcome. 

Installation
------------

Use the source !!!! 

Dependencies
~~~~~~~~~~~~

PyAF requires::

- scikit-learn and its dependencies.
- Pandas
- Python (>= 3.5),
- NumPy,
- SciPy.
- matplotlib
- SQLAlchemy (for code generation  , optional).

Development
-----------

Code contributions are welcome. Bu reports, request for new features and documentation, tests are welcome. Please use Github platform for these tasks.

Source code
~~~~~~~~~~~

You can check the latest sources from Github with the command::

    git clone https://github.com/antoinecarme/pyaf.git


Project history
---------------

This project was started in summer 2016 as an individual challenge to check the feasibilty of an automatic forecasting tool only based on python availabe data science software (numpy, scipy, pandas, scikit-learn etc).

The project is provided as an open source library (BSD-3 License).

See the AUTHORS.rst file for a complete list of contributors.

Help and Support
----------------

Support will be provided when possible. Bug reports, Improvement requests, Documentation, Hints and Test scripts are welcome.

Documentation
~~~~~~~~~~~~~

This ia WIP. Top priority. Please, use the sample notebooks as tutorials.

Communication
~~~~~~~~~~~~~

Comments , appreciations, remarks , etc .... are welcome. 
