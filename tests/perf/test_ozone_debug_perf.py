import pandas as pd
import numpy as np

# from memory_profiler import profile
# from memprof import *

import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds

#get_ipython().magic('matplotlib inline')


# @memprof
def test_ozone_debug_perf():
    b1 = tsds.load_ozone()
    df = b1.mPastData

    # df.tail(10)
    # df[:-10].tail()
    # df[:-10:-1]
    # df.describe()

    lEngine = autof.cForecastEngine()
    lEngine

    H = b1.mHorizon;
    lEngine.mOptions.mDebugPerformance = True;
    lEngine.mOptions.mEnableCycles = False;
    lEngine.mOptions.mEnableTimeBasedTrends = False;
    lEngine.mOptions.mEnableARModels = False;
    lEngine.train(df , b1.mTimeVar , b1.mSignalVar, H);
    lEngine.getModelInfo();
    print(lEngine.mSignalDecomposition.mTrPerfDetails.head());
    
    lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution
    
    lEngine.standrdPlots("outputs/my_ozone");
    
    dfapp_in = df.copy();
    dfapp_in.tail()

    dfapp_out = lEngine.forecast(dfapp_in, H);
    dfapp_out.to_csv("outputs/ozone_apply_out.csv")
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




test_ozone_debug_perf();
