import pandas as pd
import numpy as np
import AutoForecast as autof
import Bench.TS_datasets as tsds
import warnings

from CodeGen import TS_CodeGenerator as tscodegen

#get_ipython().magic('matplotlib inline')


with warnings.catch_warnings():
    # warnings.simplefilter("error")

    b1 = tsds.load_ozone()
    df = b1.mPastData
    
    H = b1.mHorizon;
    
    N = df.shape[0];
    for n in range(2*H,  N , 10):
        df1 = df.head(n).copy();
        lAutoF = autof.cForecastEngine()
        lAutoF
        #    lAutoF.mOptions.mEnableSeasonals = False;
        #    lAutoF.mOptions.mEnableCycles = False;
        #    lAutoF.mOptions.mEnableARModels = False;
        #    lAutoF.mOptions.mDebugCycles = True;
        lAutoF.train(df1 , b1.mTimeVar , b1.mSignalVar, H);
        lAutoF.getModelInfo();
        lAutoF.mSignalDecomposition.mBestTransformation.mTimeInfo.mResolution
        lCodeGenerator = tscodegen.cTimeSeriesCodeGenerator();
        lSQL = lCodeGenerator.testGeneration(lAutoF);
