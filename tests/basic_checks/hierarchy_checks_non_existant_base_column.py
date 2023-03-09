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
df1 = b1.mHierarchy['Data']
print(df1.tail())
cols = df1.columns
b1.mHierarchy['Data'] = pd.concat((df1, pd.DataFrame([['Unknown' , 'Other_State' , 'Australia']], columns=cols))).reset_index(drop=True);
# b1.mHierarchy['Data'] = df1.append(pd.DataFrame([['Unknown' , 'Other_State' , 'Australia']], columns=cols)).reset_index(drop=True);
print(b1.mHierarchy['Data'].tail())
train_and_force_fail(b1 , "PYAF_ERROR_HIERARCHY_BASE_COLUMN_NOT_FOUND Unknown")    
