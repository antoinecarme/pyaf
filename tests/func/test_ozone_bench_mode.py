from __future__ import absolute_import

import pandas as pd
import numpy as np


import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds


b1 = tsds.load_ozone()
df = b1.mPastData

lEngine = autof.cForecastEngine()

lEngine.mOptions.mAddPredictionIntervals = False
lEngine.mOptions.mParallelMode = False
lEngine.mOptions.set_active_transformations(['None', 'Difference' , 'Anscombe'])
lEngine.mOptions.mMaxAROrder = 16

lEngine

H = b1.mHorizon;
# lEngine.mOptions.enable_slow_mode();
# lEngine.mOptions.mDebugPerformance = True;
lEngine.train(df , b1.mTimeVar , b1.mSignalVar, H);
lEngine.getModelInfo();

lEngine.standardPlots("outputs/func_ozone_bench_mode")

dfapp_in = df;
# dfapp_in.tail()

#H = 12
dfapp_out = lEngine.forecast(dfapp_in, H);
#dfapp_out.to_csv("outputs/ozone_apply_out.csv")
dfapp_out.tail(2 * H)
print("Forecast Columns " , dfapp_out.columns);
Forecast_DF = dfapp_out[[b1.mTimeVar , b1.mSignalVar, b1.mSignalVar + '_Forecast']]
# print(Forecast_DF.info())
print("Forecasts\n" , Forecast_DF.tail(H));


