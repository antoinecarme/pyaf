from __future__ import absolute_import

import pandas as pd
import numpy as np


import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds


b1 = tsds.load_ozone()
df = b1.mPastData

#df.tail(10)
#df[:-10].tail()
#df[:-10:-1]
#df.describe()


lEngine = autof.cForecastEngine()
lEngine

H = b1.mHorizon;
df[b1.mSignalVar + "_version_1"] = df[b1.mSignalVar]
df[b1.mSignalVar + "_version_2"] = df[b1.mSignalVar] / 2
df[b1.mSignalVar + "_version_3"] = df[b1.mSignalVar] / 3
df[b1.mSignalVar + "_version_4"] = df[b1.mSignalVar] / 4
lSignals = [x for x in df.columns if x.startswith(b1.mSignalVar + "_")]

lHorizons = {}
for (i, sig) in enumerate(lSignals):
    lHorizons[sig] = i * 3 + 3

# lEngine.mOptions.enable_slow_mode();
# lEngine.mOptions.mDebugPerformance = True;
lEngine.train(df , b1.mTimeVar , lSignals, lHorizons);
lEngine.getModelInfo();


lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution

lEngine.standardPlots("outputs/my_ozone");

dfapp_in = df.copy();
dfapp_in.tail()

#H = 12
dfapp_out = lEngine.forecast(dfapp_in, lHorizons);
#dfapp_out.to_csv("outputs/ozone_apply_out.csv")
dfapp_out.tail(2 * H)
print("Forecast Columns " , dfapp_out.columns);
lForcasts = [lSignal + "_Forecast" for lSignal in lSignals]
Forecast_DF = dfapp_out[[b1.mTimeVar] + lSignals + lForcasts]
print(Forecast_DF.info())
print("Forecasts\n" , Forecast_DF.tail(H));

print("\n\n<ModelInfo>")
print(lEngine.to_json());
print("</ModelInfo>\n\n")
print("\n\n<Forecast>")
print(Forecast_DF.tail(2*H).to_json(date_format='iso'))
print("</Forecast>\n\n")

