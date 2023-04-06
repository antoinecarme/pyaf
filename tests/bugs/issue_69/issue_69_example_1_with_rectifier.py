import numpy as np
import pandas as pd

# %matplotlib inline

np.random.seed(1789)

N = 3600
lRand = 0.1 * np.random.randn(N)
df_train = pd.DataFrame({"Date" : pd.date_range(start="2016-01-25", periods=N, freq='D'),
                         "Signal" : np.sin(2* np.pi * 20 * np.arange(N)/N)})

# keep the trend at the beginning of the signal, remove negative values at the end
df_train['Signal'] = df_train['Signal'].apply(lambda x : max(x,0)) + lRand * lRand


import pyaf.ForecastEngine as autof
lEngine = autof.cForecastEngine()
lEngine.mOptions.set_active_transformations([])
lEngine.mOptions.set_active_trends(['LinearTrend' , 'PolyTrend'])
lEngine.mOptions.mForecastRectifier = "relu"
lEngine.train(iInputDS = df_train, iTime = 'Date', iSignal = 'Signal', iHorizon = 7);
lEngine.getModelInfo() # => relative error 7% (MAPE)

lEngine.standardPlots("outputs/issue_69_Signal_with_rectifier")

df_forecast = lEngine.forecast(iInputDS = df_train, iHorizon = 7)
min_forecast = df_forecast['Signal_Forecast'].min()
print("MIN__FORECAST" , min_forecast)
assert(min_forecast >= 0)

