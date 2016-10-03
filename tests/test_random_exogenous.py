import pandas as pd
import numpy as np
import AutoForecast.ForecastEngine as autof
import AutoForecast.Bench.TS_datasets as tsds

import AutoForecast.CodeGen.TS_CodeGenerator as tscodegen

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("error")

    b1 = tsds.generate_random_TS(N = 600 , FREQ = 'D', seed = 0, trendtype = "constant", cycle_length = 12, transform = "", sigma = 0.0, exog_count = 2000);
    df = b1.mPastData

    # this script works on mysql with N = 600, exog_count = 20 when thread_stack = 1920K in
    # /etc/mysql/mysql.conf.d/mysqld.cnf

    df.to_csv("outputs/rand_exogenous.csv")
    
    H = b1.mHorizon;
    
    N = df.shape[0];
    for n in [N//4 , N//2 ,  N]:
        for nbex in [10 , 20 , 40 , 80, 160, 320, 640, 1280]:
            df1 = df.head(n).copy();
            lEngine = autof.cForecastEngine()
            # lEngine.mOptions.mEnableSeasonals = False;
            # lEngine.mOptions.mDebugCycles = True;
            lEngine
            lExogenousData = (b1.mExogenousDataFrame , b1.mExogenousVariables[0:nbex]) 
            lEngine.train(df1 , b1.mTimeVar , b1.mSignalVar, H, lExogenousData);
            lEngine.getModelInfo();
            lEngine.standrdPlots(name = "outputs/my_exog_" + str(nbex) + "_" + str(n));
            lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution

            dfapp_in = df1.copy();
            dfapp_in.tail()


            dfapp_out = lEngine.forecast(dfapp_in, H);
            dfapp_out.tail(2 * H)
            print("Forecast Columns " , dfapp_out.columns);
            Forecast_DF = dfapp_out[[b1.mTimeVar , b1.mSignalVar, b1.mSignalVar + '_Forecast']]
            print(Forecast_DF.info())
            print("Forecasts\n" , Forecast_DF.tail(H).values);

            print("\n\n<ModelInfo>")
            print(lEngine.to_json());
            print("</ModelInfo>\n\n")
            print("\n\n<Forecast>")
            print(Forecast_DF.to_json(date_format='iso'))
            print("</Forecast>\n\n")
