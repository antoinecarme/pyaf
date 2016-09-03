import pandas as pd
import numpy as np
import AutoForecast as autof
import Bench.TS_datasets as tsds

#get_ipython().magic('matplotlib inline')

b1 = tsds.load_ozone()
df = b1.mPastData

#df.tail(10)
#df[:-10].tail()
#df[:-10:-1]
#df.describe()


lAutoF = autof.cAutoForecast()
lAutoF

H = b1.mHorizon;
lAutoF.train(df , b1.mTimeVar , b1.mSignalVar, H);
lAutoF.getModelInfo();

lAutoF.mSignalDecomposition.mBestTransformation.mTimeInfo.mResolution

lAutoF.standrdPlots("my_ozone");

dfapp_in = df.copy();
dfapp_in.tail()

#H = 12
dfapp_out = lAutoF.forecast(dfapp_in, H);
dfapp_out.tail(2 * H)
print("Forecast Columns " , dfapp_out.columns);
Forecast_DF = dfapp_out[[b1.mTimeVar , b1.mSignalVar, b1.mSignalVar + '_BestModelForecast']]
print(Forecast_DF.info())
print("Forecasts\n" , Forecast_DF.tail(H).values);

print("\n\n<ModelInfo>")
print(lAutoF.to_json());
print("</ModelInfo>\n\n")
print("\n\n<Forecast>")
print(Forecast_DF.to_json(date_format='iso'))
print("</Forecast>\n\n")

