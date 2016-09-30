import pandas as pd
import numpy as np

import AutoForecast.ForecastEngine as autof
import AutoForecast.Bench.TS_datasets as tsds

import AutoForecast.CodeGen.TS_CodeGenerator as tscodegen

import cProfile

#get_ipython().magic('matplotlib inline')

b1 = tsds.load_ozone_exogenous()
df = b1.mPastData

#df.tail(10)
#df[:-10].tail()
#df[:-10:-1]
#df.describe()



H = b1.mHorizon;

N = df.shape[0];
for n in [N]:
    df1 = df.head(n);
    lEngine = autof.cForecastEngine()
    lEngine
    #    lEngine.mOptions.mEnableSeasonals = False;
    #    lEngine.mOptions.mEnableCycles = False;
    #    lEngine.mOptions.mEnableARModels = False;
    #    lEngine.mOptions.mDebugCycles = True;
    cProfile.run(lEngine.train(df1 , b1.mTimeVar , b1.mSignalVar, H, b1.mExogenousVariables));
    lEngine.getModelInfo();
    # lEngine.mSignalDecomposition.mBestTransformation.mTimeInfo.mResolution
    # lEngine.standrdPlots(name = "my_arx_ozone_" + str(n))
