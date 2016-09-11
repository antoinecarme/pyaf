import pandas as pd
import numpy as np
import AutoForecast as autof
import Bench.TS_datasets as tsds

from CodeGen import TS_CodeGenerator as tscodegen

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("error")

    b1 = tsds.generate_random_TS(N = 600 , FREQ = 'D', seed = 0, trendtype = "constant", cycle_length = 12, transform = "", sigma = 0.0, exog_count = 2000);
    df = b1.mPastData

    # this script works on mysql with N = 600, exog_count = 20 when thread_stack = 1920K in
    # /etc/mysql/mysql.conf.d/mysqld.cnf

    df.to_csv("rand_exogenous.csv")
    
    H = b1.mHorizon;
    
    N = df.shape[0];
    for nbex in range(0, 2000, 10):
        for n in range(4*H,  N , 10):
            df1 = df.head(n).copy();
            lAutoF = autof.cAutoForecast()
            # lAutoF.mOptions.mEnableSeasonals = False;
            # lAutoF.mOptions.mDebugCycles = True;
            lAutoF
            lAutoF.train(df1 , b1.mTimeVar , b1.mSignalVar, H, b1.mExogenousVariables[0:nbex]);
            lAutoF.getModelInfo();
            lAutoF.standrdPlots(name = "my_exog_" + str(n));
            lAutoF.mSignalDecomposition.mBestTransformation.mTimeInfo.mResolution

            dfapp_in = df1.copy();
            dfapp_in.tail()


            dfapp_out = lAutoF.forecast(dfapp_in, H);
            dfapp_out.tail(2 * H)
            print("Forecast Columns " , dfapp_out.columns);
            Forecast_DF = dfapp_out[[b1.mTimeVar , b1.mSignalVar, b1.mSignalVar + '_BestModelForecast']]
            print(Forecast_DF.info())
            print("Forecasts\n" , Forecast_DF.tail(H).values);

            print("\n\n<ModelInfo>")
            print(lAutoF.to_json());
            print("</ModelInfo>\n\n")
            print("\n\n<Forecast>")
            print(Forecast_DF.to_json(date_format='iso'))
            print("</Forecast>\n\n")
