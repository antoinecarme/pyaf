import pandas as pd
import numpy as np
import pyaf.HierarchicalForecastEngine as hautof
import pyaf.Bench.TS_datasets as tsds

import datetime

#get_ipython().magic('matplotlib inline')

b1 = tsds.load_AU_hierarchical_dataset();
df = b1.mPastData;

lEngine = hautof.cHierarchicalForecastEngine()
lEngine.mOptions.mHierarchicalCombinationMethod = "OC";
lEngine.mOptions.set_active_autoregressions([]);
lEngine

H = b1.mHorizon;

# lEngine.mOptions.enable_slow_mode();
# lEngine.mOptions.mDebugPerformance = True;
lEngine.train(df , b1.mTimeVar , b1.mSignalVar, H, b1.mHierarchy, None);

lEngine.getModelInfo();
lEngine.mSignalHierarchy.plot("outputs/test_hierarchy_AU_OC");

dfapp_in = df.copy();
dfapp_in.tail()

dfapp_out = lEngine.forecast(dfapp_in, H);
#dfapp_out.to_csv("outputs/Hierarchical_AU_apply_out.csv")
