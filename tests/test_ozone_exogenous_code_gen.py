import pandas as pd
import numpy as np
import AutoForecast as autof
import Bench.TS_datasets as tsds
import TS_CodeGenerator as tscodegen

#get_ipython().magic('matplotlib inline')

b1 = tsds.load_ozone_exogenous()
df = b1.mPastData

#df.tail(10)
#df[:-10].tail()
#df[:-10:-1]
#df.describe()



H = b1.mHorizon;

N = df.shape[0];
for n in range(2*H,  N , 10):
    df1 = df.head(n).copy();
    lAutoF = autof.cAutoForecast()
    lAutoF
#    lAutoF.mOptions.mEnableSeasonals = False;
#    lAutoF.mOptions.mEnableCycles = False;
#    lAutoF.mOptions.mEnableARModels = False;
#    lAutoF.mOptions.mDebugCycles = True;
    lAutoF.train(df1 , b1.mTimeVar , b1.mSignalVar, H, b1.mExogenousVariables);
    lAutoF.getModelInfo();
    lAutoF.mSignalDecomposition.mBestTransformation.mTimeInfo.mResolution
    lAutoF.standrdPlots(name = "my_arx_ozone_")
    lCodeGenerator = tscodegen.cTimeSeriesCodeGenerator();
    lSQL = lCodeGenerator.testGeneration(lAutoF);
