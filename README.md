
PyAF (Python Automatic Forecasting)
===================================

[![Build Status](https://travis-ci.org/antoinecarme/pyaf.svg?branch=master)](https://travis-ci.org/antoinecarme/pyaf)

PyAF is an Open Source Python library for Automatic Forecasting built on top of
popular data science python modules : numpy, scipy, pandas and scikit-learn.

PyAF works as an automated process for predicting future values of a signal
using a machine learning approach. It provides a set of features that is
comparable to some popular commercial automatic forecasting products.

PyAF has been developed, tested and benchmarked using a **python 3.x** version.

PyAF is distributed under the [3-Clause BSD license](https://tldrlegal.com/license/bsd-3-clause-license-%28revised%29).


Demo 
----
[also availabe as a jupyter notebook](docs/sample_code.ipynb)

```Python
import numpy as np
import pandas as pd

# generate a daily signal covering one year 2016 in a pandas dataframe
N = 360
df_train = pd.DataFrame({"Date" : pd.date_range(start="2016-01-25", periods=N, freq='D'),
                         "Signal" : (np.arange(N)//40 + np.arange(N) % 21 + np.random.randn(N))})

import pyaf.ForecastEngine as autof
# create a forecast engine. This is the main object handling all the operations
lEngine = autof.cForecastEngine()

# get the best time series model for predicting one week
lEngine.train(iInputDS = df_train, iTime = 'Date', iSignal = 'Signal', iHorizon = 7);
lEngine.getModelInfo() # => relative error 7% (MAPE)

# predict one week
df_forecast = lEngine.forecast(iInputDS = df_train, iHorizon = 7)
# list the columns of the forecast dataset
print(df_forecast.columns) #

# print the real forecasts
# Future dates : ['2017-01-19T00:00:00.000000000' '2017-01-20T00:00:00.000000000' '2017-01-21T00:00:00.000000000' '2017-01-22T00:00:00.000000000' '2017-01-23T00:00:00.000000000' '2017-01-24T00:00:00.000000000' '2017-01-25T00:00:00.000000000']
print(df_forecast['Date'].tail(7).values)

# signal forecast : [ 9.74934646  10.04419761  12.15136455  12.20369717  14.09607727 15.68086323  16.22296559]
print(df_forecast['Signal_Forecast'].tail(7).values)
```

Features
--------
PyAF allows forecasting a time series (or a signal) for future values in a fully automated
way. To build forecasts, PyAF allows using time information (by identifying **long-term** evolution and **periodic patterns**), analyzes the **past of the signal**, exploits **exogenous data** (user-provided time series that may be correlated with the signal) as well as the **hierarchical structure** of the signal (by aggregating spatial components forecasts, for example) 

PyAF **uses [Pandas](http://pandas.pydata.org/) as a data access layer**. It consumes data coming from a pandas data-
frame (with time and signal columns), builds a time series model, and outputs
the forecasts in a pandas data-frame. Pandas is an excellent data access layer,
it allows reading/writing a huge set of file formats, accessing various data
sources (databases) and has an extensive set of algorithms to handle data-
frames (aggregation, statistics, linear algebra, plotting etc).


PyAF statistical time series models are built/estimated/trained using [scikit-learn library](http://scikit-learn.org).


The following features are available :
   1. **Training a model** to forecast a time series (given in a pandas data-frame
      with time and signal columns).
        * PyAF uses a **machine learning approach** (The signal is cut into Estimation
      and validation parts, respectively, 80% and 20% of the signal).
   2. Forecasting a time series model on a given **horizon** (forecast result is
      also pandas data-frame) and providing **prediction/confidence intervals** for
      the forecasts.
   3. Generic training features
         * [Signal decomposition](http://en.wikipedia.org/wiki/Decomposition_of_time_series) as the sum of a trend, periodic and AR component
         * PyAF works as a competition between a **comprehensive set of possible signal 
      transformations and linear decompositions**. For each transformed
      signal , a set of possible trends, periodic components and AR models is
      generated and all the possible combinations are estimated. The best
      decomposition in term of performance is kept to forecast the signal (the
      performance is computed on a part of the signal that was not used for the
      estimation).
         * **Signal transformation** is supported before **signal decompositions**. Four
      transformations are supported by default. Other transformation are
      available (Box-Cox etc).
         * All Models are estimated using **standard procedures and state-of-the-art
      time series modeling**. For example, trend regressions and AR/ARX models
      are estimated using scikit-learn linear regression models.
      * Standard performance measures are used (L1, RMSE, MAPE, etc)
   4. **Exogenous Data Support**
        * Exogenous data can be provided to improve the forecasts. These are
      expected to be **stored in an external data-frame** (this data-frame will be
      merged with the training data-frame).
        * Exogenous data are integrated in the modeling process through their **past values**
      ([ARX models](http://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model)).
        * **Exogenous variables can be of any type** (numeric, string , date, or
      object).
        * Exogenous variables are **dummified** for the non-numeric types, and
      **standardized** for the numeric types.
   5. PyAF implements **Hierarchical Forecasting**. It follows the excellent approach used in [Rob J
      Hyndman and George Athanasopoulos book](http://www.otexts.org/fpp/9/4). Thanks @robjhyndman
        * **Hierarchies** and **grouped time series** are supported.
        * **Bottom-Up**, **Top-Down** (using proportions), **Middle-Out** and **Optimal Combinations** are
      implemented.
   6. The modeling process is **customizable** and has a huge set of **options**. The
      default values of these options should however be OK.
   7. A **benchmarking process** is in place (using M1, M2, M3 competitions, NN3,
      NN5 forecasting competitions).
         * This process will be used to control the quality of modeling changes introduced in the future versions of PyAF. A  [related  github issue](https://github.com/antoinecarme/pyaf/issues/45) is created. 
         * Benchmarks data/reports are saved in a separate [github repository](https://github.com/antoinecarme/PyAF_Benchmarks). 
         * Sample [benchmark report](https://github.com/antoinecarme/PyAF_Benchmarks/blob/master/reporting/data/M1_COMP_debrief.csv) with 1001 datasets from the M1 Forecasting Competition.
   8. Basic **plotting** functions using matplotlib with standard time series and
      forecasts plots.
   9. **Software Quality** Highlights
      * An **object-oriented** approach is used for the system design. Separation of
      concerns is the key factor here.
      * **Fully written in python** with numpy, scipy, pandas and scikit-learn
      objects. Tries to be **column-based** everywhere for performance reasons.
      * Internally using a fit/predict pattern , inspired by scikit-learn, to estimate/forecast the different signal components (trends, cycles and AR models).
      * A **test-driven approach** is used. Test scripts are available in the [tests](tests)
      directory, one directory for each feature.
      -Some **[jupyter notebooks](docs)** are available for demo purposes with standard time series and forecasts plots.
      * Very **simple API** for training and forecasting.
   10. A basic **RESTful Web Service** (Flask) is available.
       * This service allows building a time series model, forecasting future data and some standard plots by providing a minimal specification of the signal in the JSON request body (at least a link to a csv file containing the data).
       * See [this doc](WS/README.md) and the [related github issue](https://github.com/antoinecarme/pyaf/issues/20) for more details.
   11. A project for **SQL generation** is started (using core **[SQLAlchemy](http://www.sqlalchemy.org/)** expressions). 
        The goal is to be able to export the forecasts as a SQL
        code to ease the **production mode**. SQLAlchemy provides **agnostic support of
        a large set of databases**.

PyAF is a work in progress. The set of features is evolving. Your feature
requests, comments, help, hints are very welcome.


Installation
------------

PyAF has been developed, tested and used on a python 3.x version. 

The following two commands install PyAF and all its dependencies:

	pip install scipy pandas sklearn matplotlib pydot dill pathos sqlalchemy
	pip install --upgrade git+git://github.com/antoinecarme/pyaf.git

Development
-----------

Code contributions are welcome. Bug reports, request for new features and
documentation, tests are welcome. Please use GitHub platform for these tasks.

You can check the latest sources of PyAF from GitHub with the command::

	git clone http://github.com/antoinecarme/pyaf.git


Project history
-----------

This project was started in summer 2016 as a POC to check the feasibility of an
automatic forecasting tool based only on python available data science software
(numpy, scipy, pandas, scikit-learn etc).

See the [AUTHORS.rst](AUTHORS.rst) file for a complete list of contributors.

Help and Support
----------------

PyAF is currently maintained by the original developer. PyAF support will be
provided when possible.

Bug reports, Improvement requests, Documentation, Hints and Test scripts are
welcome. Please use the GitHub platform for these tasks.

Documentation
----------------

An [introductory notebook](https://github.com/antoinecarme/pyaf/blob/master/docs/PyAF_Introduction.ipynb) to the time series forecasting with PyAF is available here. It contains some real-world examples and use cases.

A specific notebook describing the use of exogenous data is [available here](https://github.com/antoinecarme/pyaf/blob/master/docs/PyAF_Exogenous.ipynb)

Notebooks describing an example of hierarchical forecasting models are available for [Signal Hierarchies](https://github.com/antoinecarme/pyaf/blob/master/docs/PyAF_Hierarchical_FrenchWineExportation.ipynb) and for [Grouped Signals](https://github.com/antoinecarme/pyaf/blob/master/docs/PyAF_GroupedSignals_FrenchWineExportation.ipynb)

The python code is not yet fully documented. This is a top priority (TODO). 

Communication
----------------

Comments , appreciations, remarks , etc .... are welcome. Your feedback is
welcome if you use this library in a project or a publication.
