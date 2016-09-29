import pandas as pd
import numpy as np
import AutoForecast as autof
import Bench.TS_datasets as tsds
import TS_CodeGen_Objects as tscodegen

#get_ipython().magic('matplotlib inline')

b1 = tsds.generate_random_TS(N = 3200 , FREQ = 'D', seed = 0, trendtype = "linear", cycle_length = 12, transform = "", sigma = 0.0);
df = b1.mPastData
df.to_csv("acfrefefs_cycle.csv")
#df.tail(10)
#df[:-10].tail()
#df[:-10:-1]
#df.describe()

H = b1.mHorizon;

N = df.shape[0];
for n in range(24*H,  N , 10):
    df1 = df.head(n).copy();
    lAutoF = autof.cForecastEngine()
    # lAutoF.mOptions.mEnableSeasonals = False;
    # lAutoF.mOptions.mDebugCycles = True;
    lAutoF
    lAutoF.train(df1 , b1.mTimeVar , b1.mSignalVar, H);
    lAutoF.getModelInfo();
    # lAutoF.standrdPlots(name = "my_cycle_" + str(n));
    lAutoF.mSignalDecomposition.mBestTransformation.mTimeInfo.mResolution
    lCodeGenerator = tscodegen.cDecompositionCodeGenObject();
    lSQL = lCodeGenerator.testGeneration(lAutoF);
