
def getVersions():
        
    import os, platform
    lVersionDict = {};
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

    import sqlalchemy
    lVersionDict["sqlalchemy_version"] = sqlalchemy.__version__
    
    print([(k, lVersionDict[k]) for k in sorted(lVersionDict)]);
    return lVersionDict;
    


getVersions()
