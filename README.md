
Author: Antoine Carme <antoine.carme@laposte.net>
License: BSD 3 clause

[![Build Status](https://travis-ci.org/antoinecarme/pyaf.svg?branch=master)](https://travis-ci.org/antoinecarme/pyaf)

PyAF (Python Automatic Forecasting)
===================================

PyAF is an Open Source Python library for Automatic Forecasting built on top of
popular data science python modules : numpy, scipy, pandas and scikit-learn.

PyAF works as an automated process for predicting future values of a signal
using a machine learning approach. It provides a set of features that is
comparable to some popular commercial automatic forecasting products.

PyAF is distributed under the 3-Clause BSD license.

Features
--------
PyAF allows forecasting a time series for future values in a fully automated
way.


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
   7. A **benchmarking process** is progressing (using M1, M2, M3 competitions, NN3,
      NN5 data).
   8. Basic **plotting** functions using matplotlib with standard time series and
      forecasts plots.
   9. **Software Quality** Highlights
    * An **object-oriented** approach is used for the system design. Separation of
      concerns is the key factor here.
    * **Fully written in python** with numpy, scipy, pandas and scikit-learn
      objects. Tries to be **column-based** everywhere for performance reasons.
    * A **test-driven approach** is used. Test scripts are available in the [tests](tests)
      directory, one directory for each feature.
      -Some **[jupyter notebooks](docs)** are available for demo purposes with standard time series and forecasts plots.
    * Very **simple API** for training and forecasting.
   10. A basic **REST WebService** (Flask) effort is starting.
    * See http://pyaf.herokuapp.com/ and the [related github issue](https://github.com/antoinecarme/pyaf/issues/20)
   11. A project for **SQL generation** is started (using core **[SLQAlchemy](http://www.sqlalchemy.org/)** expressions). 
        The goal is to be able to export the forecasts as a SQL
        code to ease the **production mode**. SLQAlchemy provides **agnostic support of
        a large set of databases**.

PyAF is a work in progress. The set of features is evolving. Your feature
requests, comments, help, hints are very welcome.


Installation
------------

Use the source !!!!

No package is available yet. It is however easy to colne the repository in a directory called 'pyaf' using the command :

	git clone http://github.com/antoinecarme/pyaf.git

and add the parent directory of 'pyaf' to your PYTHONPATH environment variable, in a bash shell, this can be done with:
        
	export PYTHONPATH=$PYTHONPATH:full_path_to_parent_dir_of_pyaf

Dependencies
------------

PyAF requires::

	- scikit-learn.
	- Pandas
	- Python (tested with 3.5),
	- NumPy,
	- SciPy.
	- matplotlib
	- SQLAlchemy (for code generation , optional).

Development
-----------

Code contributions are welcome. Bug reports, request for new features and
documentation, tests are welcome. Please use Github platform for these tasks.

Source code
-----------

You can check the latest sources of PyAF from Github with the command::

	git clone http://github.com/antoinecarme/pyaf.git


Project history
-----------

This project was started in summer 2016 as a POC to check the feasibility of an
automatic forecasting tool based only on python available data science software
(numpy, scipy, pandas, scikit-learn etc).

PyAF is provided as an open source library (BSD-3 License).

See the [AUTHORS.rst](AUTHORS.rst) file for a complete list of contributors.

Help and Support
----------------

PyAF is currently maintained by the original developer. PyAF support will be
provided when possible.

Bug reports, Improvement requests, Documentation, Hints and Test scripts are
welcome. Please use the Github platform for these tasks.

Documentation
----------------

An [introductory notebook](https://github.com/antoinecarme/pyaf/blob/master/docs/PyAF_Introduction.ipynb) to the time series forecasting with PyAF is available here. It contains some real-world examples and use cases.

A specific notebook describing the use of exogenous data is [available here](https://github.com/antoinecarme/pyaf/blob/master/docs/PyAF_Exogenous.ipynb)

A notebook describing an example of hierarchical forecasting models is [available here](https://github.com/antoinecarme/pyaf/blob/master/docs/PyAF_Hierarchical_FrenchWineExportation.ipynb)

The python code is not yet fully documented. This is a top priority (TODO). 

Communication
----------------

Comments , appreciations, remarks , etc .... are welcome. Your feedback is
welcome if you use this library in a project or a publication.
