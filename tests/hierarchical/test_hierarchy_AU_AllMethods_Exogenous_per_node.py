import pandas as pd
import numpy as np
import pyaf.HierarchicalForecastEngine as hautof
import pyaf.Bench.TS_datasets as tsds

import datetime

#get_ipython().magic('matplotlib inline')

def create_exog_data(b1):
    # fake exog data based on date variable
    lDate1 = b1.mPastData['Date']
    lDate2 = b1.mFutureData['Date'] # not needed. exogfenous data are missing when not available.
    lDate = pd.concat((lDate1, lDate2), axis = 0)
    lExogenousDataFrame = pd.DataFrame()
    lExogenousDataFrame['Date'] = lDate
    lExogenousDataFrame['Date_second'] = lDate.dt.second
    lExogenousDataFrame['Date_minute'] = lDate.dt.minute
    lExogenousDataFrame['Date_hour'] = lDate.dt.hour
    lExogenousDataFrame['Date_dayofweek'] = lDate.dt.dayofweek
    lExogenousDataFrame['Date_day'] = lDate.dt.day
    lExogenousDataFrame['Date_dayofyear'] = lDate.dt.dayofyear
    lExogenousDataFrame['Date_month'] = lDate.dt.month
    lExogenousDataFrame['Date_week'] = lDate.dt.isocalendar().week
    # a column in the exog data can be of any type
    lExogenousDataFrame['Date_day_name'] = lDate.dt.day_name()
    lExogenousDataFrame['Date_month_name'] = lDate.dt.month_name()
    lExogenousVariables = [col for col in lExogenousDataFrame.columns if col.startswith('Date_')]
    lExogenousData = {}
    # define exog only for three state nodes
    lExogenousData["NSW_State"] = (lExogenousDataFrame , lExogenousVariables[:3]) 
    lExogenousData["VIC_State"] = (lExogenousDataFrame , lExogenousVariables[-3:]) 
    lExogenousData["QLD_State"] = (lExogenousDataFrame , lExogenousVariables) 
    return lExogenousData


b1 = tsds.load_AU_hierarchical_dataset();
df = b1.mPastData;

lEngine = hautof.cHierarchicalForecastEngine()
lEngine.mOptions.mHierarchicalCombinationMethod = ["BU" , 'TD' , 'MO' , 'OC'];
lEngine.mOptions.mNbCores = 16
lEngine


H = b1.mHorizon;

# lEngine.mOptions.enable_slow_mode();
# lEngine.mOptions.mDebugPerformance = True;

lExogenousData = create_exog_data(b1)
lEngine.train(df , b1.mTimeVar , b1.mSignalVar, H, b1.mHierarchy, iExogenousData = lExogenousData);

lEngine.getModelInfo();
lEngine.mSignalHierarchy.plot("outputs/test_hierarchy_AU_AllMethods_Exogenous_per_node");

dfapp_in = df.copy();
dfapp_in.tail()

dfapp_out = lEngine.forecast(dfapp_in, H);
#dfapp_out.to_csv("outputs/Hierarchical_AU_apply_out.csv")
