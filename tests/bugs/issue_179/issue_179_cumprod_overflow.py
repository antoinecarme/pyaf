from __future__ import absolute_import

import pandas as pd
import numpy as np


# import warnings
# warnings.simplefilter("error")
    
import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds


np.random.seed(1789)

df = pd.DataFrame()
N = 1000
lTimeVar = 'Time'
lSignalVar = 'Signal'
df[lTimeVar] = pd.date_range("2018-01-01", periods=N, freq="H")
df[lSignalVar] = np.random.random(df.shape[0]) 
df.info()
print(df.head())


lEngine = autof.cForecastEngine()
lEngine

H = 120
# lEngine.mOptions.enable_slow_mode();
# lEngine.mOptions.mDebugPerformance = True;
lEngine.train(df , lTimeVar , lSignalVar, H);
lEngine.getModelInfo();


lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution

lEngine.standardPlots("outputs/noise_");

dfapp_in = df.copy();
dfapp_in.tail()

#H = 12
dfapp_out = lEngine.forecast(dfapp_in, H);
#dfapp_out.to_csv("outputs/ozone_apply_out.csv")
dfapp_out.tail(2 * H)
print("Forecast Columns " , dfapp_out.columns);
Forecast_DF = dfapp_out[[lTimeVar , lSignalVar, lSignalVar + '_Forecast']]
print(Forecast_DF.info())
print("Forecasts\n" , Forecast_DF.tail(H));

print("\n\n<ModelInfo>")
print(lEngine.to_json());
print("</ModelInfo>\n\n")
print("\n\n<Forecast>")
print(Forecast_DF.tail(2*H).to_json(date_format='iso'))
print("</Forecast>\n\n")

