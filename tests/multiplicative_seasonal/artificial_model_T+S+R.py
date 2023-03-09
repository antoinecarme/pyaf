import numpy as np
import pandas as pd

# generate a daily signal covering one year 2016 in a pandas dataframe
N = 365
np.random.seed(seed=1960)

df_train = pd.DataFrame({"Date" : pd.date_range(start="2016-01-25", periods=N, freq='D'),
                         "idx" : np.arange(N)})
df_train["T"] = df_train["idx"] * 0.05
df_train["S"] = df_train["idx"].apply(lambda x : np.mod(x, 12))
df_train["Noise"] = 0.1 * np.random.standard_normal()
df_train["Signal_T_S_R"] = df_train["T"] + df_train["S"] + df_train["Noise"]
df_train["Signal_TS_R"] = df_train["T"] * df_train["S"] + df_train["Noise"]
df_train["Signal_TSR"] = df_train["T"] * df_train["S"] * df_train["Noise"]
df_train["Signal_Constant"] = 4.0
df_train["Signal_Zero"] = 0.0
df_train["Signal_One"] = 1.0
print(df_train.head(24))

import pyaf.ForecastEngine as autof
# create a forecast engine. This is the main object handling all the operations
lEngine = autof.cForecastEngine()
lSignal = 'Signal_T_S_R'
# lEngine.mOptions.enable_slow_mode();
lEngine.mOptions.mDebugPerformance = True;
lEngine.mOptions.set_active_decomposition_types(['T+S+R', 'TS+R', 'TSR']);

# get the best time series model for predicting one week
lEngine.train(iInputDS = df_train, iTime = 'Date', iSignal = lSignal, iHorizon = 7);
lEngine.getModelInfo() # => relative error 7% (MAPE)


# predict one week
df_forecast = lEngine.forecast(iInputDS = df_train, iHorizon = 7)
# list the columns of the forecast dataset
print(df_forecast.columns) #

# print the real forecasts
# Future dates : ['2017-01-19T00:00:00.000000000' '2017-01-20T00:00:00.000000000' '2017-01-21T00:00:00.000000000' '2017-01-22T00:00:00.000000000' '2017-01-23T00:00:00.000000000' '2017-01-24T00:00:00.000000000' '2017-01-25T00:00:00.000000000']
print(df_forecast['Date'].tail(7).values)

# signal forecast : [ 9.74934646  10.04419761  12.15136455  12.20369717  14.09607727 15.68086323  16.22296559]
print(df_forecast[lSignal + '_Forecast'].tail(7).values)
lEngine.standardPlots("outputs/artificial_" + lSignal);
