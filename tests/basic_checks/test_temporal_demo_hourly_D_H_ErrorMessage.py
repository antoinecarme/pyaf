# %matplotlib inline
import pyaf

import numpy as np
import pandas as pd

DATA_FREQ = 'H'
PERIODS = ["D" , "H"]
H = 365
N = H * 10
lDateColumn = "Date"
lSignalVar = "Signal";
START_TIME = "2001-01-25"

# generate a daily signal covering one year 2016 in a pandas dataframe
np.random.seed(seed=1960)
df_train = pd.DataFrame({lDateColumn : pd.date_range(start=START_TIME, periods=N, freq=DATA_FREQ),
                         lSignalVar : (np.arange(N)//40 + np.arange(N) % 21 + np.random.randn(N))})
# print(df_train.head(N))

lHierarchy = {};
lHierarchy['Levels'] = None;
lHierarchy['Data'] = None;
lHierarchy['Groups']= {};

lHierarchy['Periods']= PERIODS

lHierarchy['Type'] = "Temporal";

import pyaf.HierarchicalForecastEngine as hautof

lEngine = hautof.cHierarchicalForecastEngine()
# lEngine.mOptions.mNbCores = 1
lEngine.mOptions.mHierarchicalCombinationMethod = ["BU" , 'TD' , 'MO' , 'OC'];

lFailed = None
try:
    lSignalHierarchy = lEngine.train(df_train , lDateColumn, lSignalVar, H, lHierarchy, None);
    lFailed = False
except Exception as lEx:
    print("ERROR" , lEx.__class__, lEx)
    lFailed = True

if(not lFailed):
    raise Exception("NORMAL_BEHAVIOR_NOT_EXPECTED_SHOULD_HAVE_FAILED")
