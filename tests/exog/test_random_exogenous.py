import pandas as pd
import numpy as np
import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds


import warnings


def test_random_exogenous(n , nbex):

    with warnings.catch_warnings():
        # warnings.simplefilter("error")

        b1 = tsds.generate_random_TS(N = 600 , FREQ = 'D', seed = 0, trendtype = "LinearTrend", cycle_length = 12, transform = "", sigma = 0.0, exog_count = 2000, ar_order = 12);
        df = b1.mPastData
        df['Signal'] = df[b1.mName]
        print(df.head())
        print(b1)
        b1.mExogenousDataFrame.info()
        H = b1.mHorizon[b1.mSignalVar];
    
        N = df.shape[0];
        df1 = df.head(n).copy();
        lEngine = autof.cForecastEngine()
        # lEngine.mOptions.mEnableSeasonals = False;
        # lEngine.mOptions.mDebugCycles = False;
        lEngine.mOptions.mDebugProfile = True;
        lEngine.mOptions.mNbCores = 1
        lEngine.mOptions.set_active_autoregressions(['ARX']);
        lEngine
        lExogenousData = (b1.mExogenousDataFrame , b1.mExogenousVariables[0:nbex]) 
        lExogenousData[0].info()
        df1.info()
        lEngine.train(df1 , b1.mTimeVar , b1.mSignalVar, H, lExogenousData);
        lEngine.getModelInfo();
        lEngine.standardPlots(name = "outputs/my_exog_" + str(nbex) + "_" + str(n));
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
        print(Forecast_DF.tail(2*H).to_json(date_format='iso'))
        print("</Forecast>\n\n")
