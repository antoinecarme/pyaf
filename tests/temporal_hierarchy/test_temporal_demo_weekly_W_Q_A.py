# %matplotlib inline
import pyaf

import numpy as np
import pandas as pd

DATA_FREQ = 'W'
PERIODS = ["W" , "Q" , "A"]
H = 36
N = H * 10
lDateColumn = "Date"
lSignalVar = "Signal";
lSignalVar = lSignalVar + "_".join(PERIODS)
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


# create a model to plot the hierarchy.
import pyaf.HierarchicalForecastEngine as hautof
lEngine = hautof.cHierarchicalForecastEngine()



lSignalHierarchy = lEngine.plot_Hierarchy(df_train , lDateColumn, lSignalVar, H, 
                                          lHierarchy, None);

# print(lSignalHierarchy.__dict__)


# create a hierarchical model and train it
import pyaf.HierarchicalForecastEngine as hautof

lEngine = hautof.cHierarchicalForecastEngine()
# lEngine.mOptions.mNbCores = 1
lEngine.mOptions.mHierarchicalCombinationMethod = ["BU" , 'TD' , 'MO' , 'OC'];

lSignalHierarchy = lEngine.train(df_train , lDateColumn, lSignalVar, H, lHierarchy, None);
lEngine.getModelInfo();

lEngine.standardPlots("outputs/temporal_demo_weekly_" + lSignalVar);


dfapp_in = df_train.copy();
dfapp_in.info()
dfapp_in.tail()

dfapp_out = lEngine.forecast(dfapp_in, H);
dfapp_out.info()
print(dfapp_out.tail())
