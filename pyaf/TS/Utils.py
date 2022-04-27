# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import os

from datetime import datetime

from functools import partial

def createDirIfNeeded(dirname):
    try:
        os.mkdir(dirname);
    except:
        pass


class cMemoize:
    def __init__(self, f):
        self.mFunction = f
        self.mCache = {}
    def __call__(self, *args):
        # print("MEMOIZING" , self.mFunction , args)
        if not args in self.mCache:
            self.mCache[args] = self.mFunction(*args)
        return self.mCache[args]
    
    def __get__(self, obj, objtype):
        # Support instance methods.
        return partial(self.__call__, obj)
  
class PyAF_Error(Exception):
    """Exception raised for errors in the forecasting process.

    Attributes:
        mReason : explanation of the error
    """

    def __init__(self, reason):
        self.mReason = reason

class Internal_PyAF_Error(PyAF_Error):
    """Exception raised for errors in the forecasting process.

    Attributes:
        mReason : explanation of the error
    """

    def __init__(self, reason):
        self.mReason = reason



def get_pyaf_logger():
    import logging;
    logger = logging.getLogger('pyaf.std');
    if(logger.handlers == []):
        import logging.config
        logging.basicConfig(level=logging.INFO)        
    return logger;

def get_pyaf_timing_logger():
    import logging;
    logger = logging.getLogger('pyaf.timing');
    if(logger.handlers == []):
        import logging.config
        logging.basicConfig(level=logging.INFO)        
    return logger;

def get_pyaf_hierarchical_logger():
    import logging;
    logger = logging.getLogger('pyaf.hierarchical');
    return logger;



class cTimer:
    def __init__(self, iMess = "PYAF_UNKNOWN_OP", iDebug = False):
        self.mMessage = iMess
        self.mStart = datetime.now();
        self.logger = get_pyaf_timing_logger();
        self.logger.info(("OPERATION_START", self.mMessage))

    def __del__(self):
        self.mEnd = datetime.now();
        lDelta = self.mEnd - self.mStart
        self.logger.info(("OPERATION_END_ELAPSED" , round(lDelta.total_seconds(), 3), self.mMessage))

def get_module_version_when_available(module_name):
    try:
        import importlib
        mod = importlib.import_module(module_name)
        return mod.__version__
    except:
        return "NOT_INSTALLED"

        
def getVersions():
        
    import platform
    lVersionDict = {};
    lVersionDict["system_platform"] = platform.platform();
    lVersionDict["system_uname"] = platform.uname();
    lVersionDict["system_processor"] = platform.processor();
    lVersionDict["python_implementation"] = platform.python_implementation();
    lVersionDict["python_version"] = platform.python_version();

    lModules = ["pyaf", "sklearn", "pandas", "numpy" , "scipy" , "matplotlib", "pydot", "sqlalchemy" ,
                "xgboost" , "keras", "pip" , "setuptools", "Cython", "dill" , "pathos", "lightgbm",
                "torch", "skorch"]

    for module_name in lModules:
        lVersionDict[module_name + "_version"] = get_module_version_when_available(module_name)
    
    # print([(k, lVersionDict[k]) for k in sorted(lVersionDict)]);
    return lVersionDict;
