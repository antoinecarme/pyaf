# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import sys, os

def createDirIfNeeded(dirname):
    try:
        os.mkdir(dirname);
    except:
        pass



class ForecastError(Exception):
    """Exception raised for errors in the forecasting process.

    Attributes:
        mReason : explanation of the error
    """

    def __init__(self, reason):
        self.mReason = reason

class InternalForecastError(Exception):
    """Exception raised for errors in the forecasting process.

    Attributes:
        mReason : explanation of the error
    """

    def __init__(self, reason):
        self.mReason = reason



def get_pyaf_logger():
    import logging;
    logger = logging.getLogger('pyaf.std');
    return logger;

def get_pyaf_hierarchical_logger():
    import logging;
    logger = logging.getLogger('pyaf.hierarchical');
    return logger;
