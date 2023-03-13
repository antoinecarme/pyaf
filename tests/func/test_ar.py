import pandas as pd
import numpy as np

import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds

#get_ipython().magic('matplotlib inline')

b1 = tsds.generate_random_TS(N = 320 , FREQ = 'D', seed = 0, trendtype = "constant", cycle_length = 0, transform = "None", sigma = 0.1, ar_order = 12);
df = b1.mPastData

df['Signal'] = df[b1.mName]

#df.tail(10)
#df[:-10].tail()
#df[:-10:-1]
#df.describe()


lEngine = autof.cForecastEngine()
lEngine
lEngine.mOptions.set_active_transformations([]);
lEngine.mOptions.set_active_trends([]);
lEngine.mOptions.set_active_periodics([]);
lEngine.mOptions.set_active_autoregressions(['AR' , 'ARX']);

H = b1.mHorizon[b1.mSignalVar];
lEngine.train(df , b1.mTimeVar , b1.mSignalVar, H);
lEngine.getModelInfo();


lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution

lEngine.standardPlots("outputs/my_random_ar");

dfapp_in = df.copy();
dfapp_in.tail()

#H = 12
dfapp_out = lEngine.forecast(dfapp_in, H);
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
