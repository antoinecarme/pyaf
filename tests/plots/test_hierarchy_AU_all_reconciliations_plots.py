import pandas as pd
import numpy as np
import pyaf.HierarchicalForecastEngine as hautof
import pyaf.Bench.TS_datasets as tsds

import datetime

#get_ipython().magic('matplotlib inline')

b1 = tsds.load_AU_hierarchical_dataset();
df = b1.mPastData;

lEngine = hautof.cHierarchicalForecastEngine()
lEngine.mOptions.mHierarchicalCombinationMethod = ["BU" , 'TD' , 'MO' , 'OC'];
lEngine.mOptions.set_active_autoregressions([]);
lEngine

H = b1.mHorizon;

# lEngine.mOptions.enable_slow_mode();
# lEngine.mOptions.mDebugPerformance = True;
lEngine.train(df , b1.mTimeVar , b1.mSignalVar, H, b1.mHierarchy, None);

lEngine.getModelInfo();
lEngine.standardPlots("outputs/plot_test_AU_all");

dfapp_in = df.copy();
dfapp_in.tail()

dfapp_out = lEngine.forecast(dfapp_in, H);
#dfapp_out.to_csv("outputs/Hierarchical_AU_apply_out.csv")


import hashlib
lDict = lEngine.getPlotsAsDict()
lDict1 = lDict["Models"]
for sig, v in lDict1.items():
    for plot_type, plot_str in v.items():
        lMD5 = hashlib.md5(plot_str.encode()).hexdigest()
        print("PLOT_PNG_DICT" , (sig, plot_type , lMD5, plot_str[:64]))

lStruct_plot_str = lDict["Hierarchical_Structure"]
lMD5_2 = hashlib.md5(lStruct_plot_str.encode()).hexdigest()
print("PLOT_PNG_DICT" , ("Hierarchical_Structure" , lMD5_2, lStruct_plot_str[:64]))
