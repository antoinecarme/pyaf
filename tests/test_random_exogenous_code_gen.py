import pandas as pd
import numpy as np
import AutoForecast as autof
import Bench.TS_datasets as tsds

from CodeGen import TS_CodeGenerator as tscodegen

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("error")

    b1 = tsds.generate_random_TS(N = 600 , FREQ = 'D', seed = 0, trendtype = "constant", cycle_length = 12, transform = "", sigma = 0.0, exog_count = 20);
    df = b1.mPastData

    # this script works on mysql with N = 600, exog_count = 20 when thread_stack = 1920K in
    # /etc/mysql/mysql.conf.d/mysqld.cnf

    df.to_csv("rand_exogenous.csv")
    
    H = b1.mHorizon;
    
    N = df.shape[0];
    for n in range(H,  N , 10):
        df1 = df.head(n).copy();
        lAutoF = autof.cForecastEngine()
        # lAutoF.mOptions.mEnableSeasonals = False;
        # lAutoF.mOptions.mDebugCycles = True;
        lAutoF
        lAutoF.train(df1 , b1.mTimeVar , b1.mSignalVar, H, b1.mExogenousVariables);
        lAutoF.getModelInfo();
        lAutoF.standrdPlots(name = "my_exog_" + str(n));
        lAutoF.mSignalDecomposition.mBestTransformation.mTimeInfo.mResolution
        lCodeGenerator = tscodegen.cTimeSeriesCodeGenerator();
        lSQL = lCodeGenerator.testGeneration(lAutoF);
