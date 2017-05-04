import pandas as pd
import numpy as np
import pyaf.HierarchicalForecastEngine as hautof
import pyaf.Bench.TS_datasets as tsds

import datetime

#get_ipython().magic('matplotlib inline')

def train_and_force_fail(b , error_message):
    try:
        df = b.mPastData;
        
        lEngine = hautof.cHierarchicalForecastEngine()
        lEngine.mOptions.mHierarchicalCombinationMethod = "TD";
        lEngine.mOptions.set_active_autoregressions([]);
        lEngine
        
        H = b.mHorizon;
        
        lEngine.train(df , b.mTimeVar , b.mSignalVar, H, b.mHierarchy, None);
        raise Exception("NOT_OK")
    except Exception as e:
        # should fail
        print(str(e));
        assert(str(e) == error_message)
        if(str(e) == "NOT_OK"):
            raise
        pass



b1 = tsds.load_AU_hierarchical_dataset();
b1.mPastData['NSW'] = "BlaBla"
train_and_force_fail(b1 , "PYAF_ERROR_HIERARCHY_BASE_SIGNAL_COLUMN_TYPE_NOT_ALLOWED 'NSW' 'object'")
