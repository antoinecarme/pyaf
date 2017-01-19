import pandas as pd
import numpy as np

import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds

import CodeGen.TS_CodeGenerator as tscodegen

stock = "BNP.PA";
b1 = tsds.load_yahoo_stock_prices("cac40")[stock]
df = b1.mPastData

df.head()
df.info();

lEngine = autof.cForecastEngine()
lEngine

lEngine.train(df , b1.mTimeVar , b1.mSignalVar , b1.mHorizon)
lEngine.getModelInfo();
print(lEngine.mSignalDecomposition.mTrPerfDetails.head());


dfapp_in = df.copy();
dfapp_in.tail()

H = b1.mHorizon
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

