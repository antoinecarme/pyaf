import pandas as pd
import numpy as np

import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds

b1 = tsds.load_ozone_exogenous_categorical()
df = b1.mPastData

print(b1.mExogenousDataFrame.Exog2.cat.categories)
print(b1.mExogenousDataFrame.Exog3.cat.categories)
print(b1.mExogenousDataFrame.Exog4.cat.categories)


lEngine = autof.cForecastEngine()
lEngine.mOptions.mDebug = True;
lEngine.mOptions.mDebugProfile = True;
lEngine.mOptions.disable_all_periodics()
lEngine.mOptions.set_active_autoregressions(['ARX'])
lExogenousData = (b1.mExogenousDataFrame , b1.mExogenousVariables) 

H = 12
lEngine.train(df , 'Time' , b1.mSignalVar, H, lExogenousData)
lEngine


lEngine.getModelInfo();
lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution
lEngine.standardPlots(name = "outputs/my_categorical_arx_ozone")

dfapp_in = df.copy();
dfapp_in.tail()

dfapp_out = lEngine.forecast(dfapp_in, H);
#dfapp_out.to_csv("outputs/arx_ozone_apply_out.csv")
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
