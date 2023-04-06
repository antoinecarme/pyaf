import pandas as pd
import numpy as np

import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds


# get_ipython().magic('matplotlib inline')
# %load_ext memory_profiler

lValues = [ 64 ];
for cyc in lValues:
    print("TEST_CYCLES_START", cyc)
    b1 = tsds.generate_random_TS(N = 4800 , FREQ = 'D', seed = 0, trendtype = "linear", cycle_length = cyc, transform = "None", sigma = 0.0, exog_count = 100, ar_order=0);
    df = b1.mPastData
    df['Signal'] = df[b1.mName]

    # df.tail(10)
    # df[:-10].tail()
    # df[:-10:-1]
    # df.describe()

    lEngine = autof.cForecastEngine()
    lEngine.mOptions.mCycleLengths = [ k for k in range(2,128) ];
    lEngine

    H = b1.mHorizon[b1.mSignalVar];
    lEngine.train(df , b1.mTimeVar , b1.mSignalVar, H);
    lEngine.getModelInfo();

    lEngine.standardPlots("outputs/perf_test_perf1_cycles_" + str(cyc));
    lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution

    dfapp_in = df.copy();
    dfapp_in.tail()

    # H = 12
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
    
    print("TEST_CYCLES_END", cyc)
