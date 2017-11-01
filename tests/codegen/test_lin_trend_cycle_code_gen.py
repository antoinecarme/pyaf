import pandas as pd
import numpy as np

import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds

import pyaf.CodeGen.TS_CodeGenerator as tscodegen

#get_ipython().magic('matplotlib inline')

b1 = tsds.generate_random_TS(N = 3200 , FREQ = 'D', seed = 0, trendtype = "linear", cycle_length = 12, transform = "", sigma = 0.0);
df = b1.mPastData
#df.to_csv("outputs/acfrefefs_cycle.csv")
#df.tail(10)
#df[:-10].tail()
#df[:-10:-1]
#df.describe()

H = b1.mHorizon;

N = df.shape[0];
for n in [N//8 , N//4 , N//2 , N]:
    df1 = df.head(n).copy();
    lEngine = autof.cForecastEngine()
    # lEngine.mOptions.mEnableSeasonals = False;
    # lEngine.mOptions.mDebugCycles = True;
    lEngine
    lEngine.train(df1 , b1.mTimeVar , b1.mSignalVar, H);
    lEngine.getModelInfo();
    # lEngine.standardPlots(name = "my_cycle_" + str(n));
    lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution
    lCodeGenerator = tscodegen.cTimeSeriesCodeGenerator();
    lSQL = lCodeGenerator.testGeneration(lEngine);
