import pandas as pd
import numpy as np
import warnings

import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds

#get_ipython().magic('matplotlib inline')
import pyaf.CodeGen.TS_CodeGenerator as tscodegen


with warnings.catch_warnings():
    # warnings.simplefilter("error")

    b1 = tsds.load_ozone()
    df = b1.mPastData
    
    H = b1.mHorizon;
    
    N = df.shape[0];
    for n in [N//4, N//2 , N]:
        df1 = df.head(n).copy();
        lEngine = autof.cForecastEngine()
        lEngine
        #    lEngine.mOptions.mEnableSeasonals = False;
        #    lEngine.mOptions.mEnableCycles = False;
        #    lEngine.mOptions.mEnableARModels = False;
        #    lEngine.mOptions.mDebugCycles = True;
        lEngine.train(df1 , b1.mTimeVar , b1.mSignalVar, H);
        lEngine.getModelInfo();
        lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution
        lCodeGenerator = tscodegen.cTimeSeriesCodeGenerator();
        lSQL = lCodeGenerator.testGeneration(lEngine);
