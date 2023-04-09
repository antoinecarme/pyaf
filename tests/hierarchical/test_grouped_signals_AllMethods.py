import pandas as pd
import numpy as np
import pyaf.HierarchicalForecastEngine as hautof
import pyaf.Bench.TS_datasets as tsds

import datetime

#get_ipython().magic('matplotlib inline')

b1 = tsds.load_AU_infant_grouped_dataset();

# reduce the number of possible values of State.
b1.mHierarchy['Groups']['State'] = ["NSW","VIC"];

df = b1.mPastData;

lEngine = hautof.cHierarchicalForecastEngine()
lEngine.mOptions.mHierarchicalCombinationMethod =  ["BU" , 'TD' , 'MO' , 'OC'];
lEngine

H = b1.mHorizon;

# lEngine.mOptions.enable_slow_mode();
# lEngine.mOptions.mDebugPerformance = True;
lEngine.mOptions.set_active_autoregressions([]);
lEngine.train(df , b1.mTimeVar , b1.mSignalVar, H, b1.mHierarchy, None);

lEngine.getModelInfo();
lEngine.mSignalHierarchy.plot("outputs/grouped_signals_AllMethods_2");

dfapp_in = df.copy();
dfapp_in.tail()

dfapp_out = lEngine.forecast(dfapp_in, H);
#dfapp_out.to_csv("outputs/Grouped_AU_apply_out.csv")


print("\n\n<ModelInfo>")
print(lEngine.to_json());
print("</ModelInfo>\n\n")
print("\n\n<Forecast>")
print(dfapp_out.tail(2*H).to_json(date_format='iso'))
print("</Forecast>\n\n")
