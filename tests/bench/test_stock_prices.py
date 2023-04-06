import pandas as pd
import numpy as np

import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds


stock = "BNP.PA";
b1 = tsds.load_yahoo_stock_prices("cac40")[stock]
df = b1.mPastData

df.head()
df.info();

lEngine = autof.cForecastEngine()
lEngine

H = b1.mHorizon[stock]

lEngine.train(df , b1.mTimeVar , b1.mSignalVar , H)
lEngine.getModelInfo();

lEngine.standardPlots("outputs/stock_prices_cac40")

dfapp_in = df.copy();
dfapp_in.tail()

dfapp_out = lEngine.forecast(dfapp_in, H);
#dfapp_out.to_csv("outputs/ozone_apply_out.csv")
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

