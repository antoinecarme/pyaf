import pandas as pd
import numpy as np
import AutoForecast.HierarchicalForecastEngine as hautof
import AutoForecast.Bench.TS_datasets as tsds

import datetime

#get_ipython().magic('matplotlib inline')

b1 = tsds.load_AU_hierarchical_dataset();
df = b1.mPastData;

lEngine = hautof.cHierarchicalForecastEngine()
lEngine

H = b1.mHorizon;

# lEngine.mOptions.enable_slow_mode();
lEngine.mOptions.mDebugPerformance = True;
lEngine.train(df , b1.mTimeVar , b1.mSignalVar, H, b1.mHierarchy, None);
lEngine.getModelInfo();

