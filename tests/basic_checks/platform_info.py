
def get_module_version_when_available(module_name):
    try:
        import importlib
        mod = importlib.import_module(module_name)
        return mod.__version__
    except:
        return "NOT_INSTALLED"
    

def getVersions():
        
    import os, platform
    lVersionDict = {};
    lVersionDict["system_platform"] = platform.platform();
    lVersionDict["system_uname"] = platform.uname();
    lVersionDict["system_processor"] = platform.processor();
    lVersionDict["python_implementation"] = platform.python_implementation();
    lVersionDict["python_version"] = platform.python_version();

    lModules = ["pyaf", "sklearn", "pandas", "numpy" , "scipy" , "matplotlib", "pydot", 
                "xgboost" , "keras", "pip" , "setuptools", "Cython", "lightgbm",
                "torch", "skorch"]

    for module_name in lModules:
        lVersionDict[module_name + "_version"] = get_module_version_when_available(module_name)
    
    # print([(k, lVersionDict[k]) for k in sorted(lVersionDict)]);
    return lVersionDict;
    


lDict = getVersions()
for k in sorted(lDict.keys()):
    print("PYAF_SYSTEM_DEPENDENT_VERSION_INFO" , (k , lDict[k]))
    

import os
lDict = os.environ
for k in sorted(lDict.keys()):
    print("PYAF_SYSTEM_DEPENDENT_ENVIRONMENT_VARIABLE" , (k , lDict[k]))

import sys
assert sys.version_info >= (3, 5)
