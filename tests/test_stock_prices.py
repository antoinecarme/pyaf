import pandas as pd
import numpy as np

import AutoForecast.ForecastEngine as autof
import AutoForecast.Bench.TS_datasets as tsds

import AutoForecast.CodeGen.TS_CodeGenerator as tscodegen


b1 = tsds.load_yahoo_stock_prices("cac40")["BNP.PA"]
df = b1.mPastData

df.head()

lDecomp = SigDec.cSignalDecomposition()
lDecomp

lDecomp.train(df , b1.mTimeVar , b1.mSignalVar , b1.mHorizon)


dfapp_in = df.copy();
dfapp_in.tail()

H = b1.mHorizon
dfapp_out = lDecomp.forecast(dfapp_in, H);
dfapp_out.tail(2 * H)
print("Forecast Columns " , dfapp_out.columns);
print("Forecasts\n" , dfapp_out.tail(H)[[b1.mTimeVar , b1.mSignalVar + '_BestModelForecast']].values);

