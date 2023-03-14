
def getVersions():
        
    import os, platform, pyaf
    lVersionDict = {};
    lVersionDict["PyAF_version"] = pyaf.__version__;
    lVersionDict["system_platform"] = platform.platform();
    lVersionDict["system_uname"] = platform.uname();
    lVersionDict["system_processor"] = platform.processor();
    lVersionDict["python_implementation"] = platform.python_implementation();
    lVersionDict["python_version"] = platform.python_version();
    
    import sklearn
    lVersionDict["sklearn_version"] = sklearn.__version__;
    
    import pandas as pd
    lVersionDict["pandas_version"] = pd.__version__;
    
    import numpy as np
    lVersionDict["numpy_version"] = np.__version__;
    
    import scipy as sc
    lVersionDict["scipy_version"] = sc.__version__;
    
    import matplotlib
    lVersionDict["matplotlib_version"] = matplotlib.__version__

    import pydot
    lVersionDict["pydot_version"] = pydot.__version__

    import xgboost
    lVersionDict["xgboost_version"] = xgboost.__version__
    
    import lightgbm
    lVersionDict["lightgbm_version"] = lightgbm.__version__
    
    import torch
    lVersionDict["torch_version"] = torch.__version__
    
    print([(k, lVersionDict[k]) for k in sorted(lVersionDict)]);
    return lVersionDict;
    


getVersions()
