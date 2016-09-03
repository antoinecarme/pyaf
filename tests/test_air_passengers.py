import pandas as pd
import numpy as np
import Bench.TS_datasets as tsds
import AutoForecast as autof

b1 = tsds.load_airline_passengers()
df = b1.mPastData

df.head()


lAutoF = autof.cAutoForecast()
lAutoF

H = b1.mHorizon;
lAutoF.train(df , b1.mTimeVar , b1.mSignalVar, H);
lAutoF.getModelInfo();

lAutoF.mSignalDecomposition.mBestTransformation.mTimeInfo.mResolution

lAutoF.standrdPlots(name = "my_airline_passengers")

dfapp_in = df.copy();
dfapp_in.tail()

#H = 12
dfapp_out = lAutoF.forecast(dfapp_in, H);
dfapp_out.tail(2 * H)
print("Forecast Columns " , dfapp_out.columns);
lForecastColumnName = b1.mSignalVar + '_BestModelForecast'
Forecast_DF = dfapp_out[[b1.mTimeVar , b1.mSignalVar, lForecastColumnName , lForecastColumnName + '_Lower_Bound',  lForecastColumnName + '_Upper_Bound' ]]
print(Forecast_DF.info())
print("Forecasts\n" , Forecast_DF.tail(2*H).values);

print("\n\n<ModelInfo>")
print(lAutoF.to_json());
print("</ModelInfo>\n\n")
print("\n\n<Forecast>")
print(Forecast_DF.to_json(date_format='iso'))
print("</Forecast>\n\n")
