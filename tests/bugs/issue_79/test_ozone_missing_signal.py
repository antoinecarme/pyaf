from __future__ import absolute_import

import pandas as pd
import numpy as np


import pyaf.ForecastEngine as autof


csvfile_link = "https://raw.githubusercontent.com/antoinecarme/pyaf/master/data/ozone-la_missing_signal.csv"
csvfile_link = "data/ozone-la_missing_signal.csv"
df = pd.read_csv(csvfile_link);
import datetime
df['Month'] = df['Month'].apply(lambda x : datetime.datetime.strptime(x, "%Y-%m"))

lEngine = autof.cForecastEngine()
lEngine

H = 12;
lTimeVar = 'Month'
lSignalVar = 'Ozone'
# lEngine.mOptions.enable_slow_mode();
lEngine.mOptions.mMissingDataOptions.mSignalMissingDataImputation = "Interpolate"

# lEngine.mOptions.mDebugPerformance = True;
lEngine.train(df , lTimeVar , lSignalVar, H);
lEngine.getModelInfo();


lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution

lEngine.standardPlots("outputs/my_ozone_missing_signal");

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

