
import numpy as np
import pandas as pd

import pyaf.ForecastEngine as autof


uri = "http://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/YahooFinance/nasdaq/yahoo_AAPL.csv"

df = pd.read_csv(uri)
df.Date = df.Date.values.astype('datetime64[D]')
df = df.sort_values(by = 'Date' , ascending=True)
df.Date.dtype


df.info()


print(df.head())

lEngine = autof.cForecastEngine()
# lEngine
lEngine.train(df , 'Date' , 'Close', 7);
lEngine.getModelInfo();

lEngine.standardPlots("outputs/yahoo_nasdaq_AAPL")

dfapp_out = lEngine.forecast(df, 7);
#dfapp_out.to_csv("outputs/ozone_apply_out.csv")
dfapp_out.tail(2 * 7)
print("Forecast Columns " , dfapp_out.columns);
Forecast_DF = dfapp_out[['Date' , 'Close', 'Close' + '_Forecast']]
Forecast_DF.info()
print("Forecasts\n" , Forecast_DF.tail(14));


