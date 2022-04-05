import pandas as pd
import numpy as np

import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds


import logging
import logging.config

#logging.config.fileConfig('logging.conf')

logging.basicConfig(level=logging.INFO)

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
    lEngine.mOptions.set_active_autoregressions(["AR", "ARX"]) # enable ARX
    #    lEngine.mOptions.mEnableSeasonals = False;
    #    lEngine.mOptions.mEnableCycles = False;
    #    lEngine.mOptions.mEnableARModels = False;
    #    lEngine.mOptions.mDebugCycles = False;
    lExogenousData = (b1.mExogenousDataFrame , b1.mExogenousVariables) 
    lEngine.train(df1 , b1.mTimeVar , b1.mSignalVar, H, lExogenousData);
    lEngine.getModelInfo();
    lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution
    # lEngine.standardPlots(name = "outputs/my_arx_ozone_" + str(n))

    dfapp_in = df1.copy();
    dfapp_in.tail()

    dfapp_out = lEngine.forecast(dfapp_in, H);
    #dfapp_out.to_csv("outputs/arx_ozone_apply_out.csv")
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

