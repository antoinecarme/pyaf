from __future__ import absolute_import

import pandas as pd
import numpy as np
import datetime

import pyaf.ForecastEngine as autof

df = pd.read_csv("data/web-traffic-time-series-forecasting/bench_4311.csv")
df['Date'] = df['Date'].apply(lambda x : datetime.datetime.strptime(x, "%Y-%m-%d"))
df = df.fillna(0.0)

#df.tail(10)
#df[:-10].tail()
#df[:-10:-1]
#df.describe()


lEngine = autof.cForecastEngine()
lEngine

lEngine.mOptions.mAddPredictionIntervals = False
lEngine.mOptions.mParallelMode = False
lEngine.mOptions.set_active_transformations(['None', 'Difference' , 'Anscombe'])
lEngine.mOptions.mMaxAROrder = 16


H = 60;
# lEngine.mOptions.enable_slow_mode();
# lEngine.mOptions.mDebugPerformance = True;

for signal in df.columns[-1:]:
    lEngine.train(df , "Date" , signal, H);
    lEngine.getModelInfo();

    lEngine.standardPlots("outputs/perf_web-traffic-time-series-forecasting_" + signal);

    lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution

    dfapp_in = df.copy();
    dfapp_in.tail()

    dfapp_out = lEngine.forecast(dfapp_in, H);
    dfapp_out.tail(2 * H)
    print("Forecast Columns " , dfapp_out.columns);
    Forecast_DF = dfapp_out[["Date" , signal, signal + '_Forecast']]
    print(Forecast_DF.info())
    print("Forecasts\n" , Forecast_DF.tail(H));

    print("\n\n<ModelInfo>")
    print(lEngine.to_json());
    print("</ModelInfo>\n\n")
    print("\n\n<Forecast>")
    print(Forecast_DF.tail(2*H).to_json(date_format='iso'))
    print("</Forecast>\n\n")

