import pandas as pd
import numpy as np
import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds
import datetime

#get_ipython().magic('matplotlib inline')

trainfile = "data/sof_example.csv";
df = pd.read_csv(trainfile, sep=r',', engine='python', skiprows=0);
df['Date'] = df['Date'].apply(lambda x : datetime.datetime.strptime(x, "%m/%d/%Y"))

print(df.head());

lDateVar = 'Date'
lSignalVar = 'Used'

lEngine = autof.cForecastEngine()
lEngine

H = 10;

#lEngine.mOptions.enable_slow_mode();
lEngine.mOptions.mDebugPerformance = True;
lEngine.train(df , lDateVar , lSignalVar, H);
lEngine.getModelInfo();
print(lEngine.mSignalDecomposition.mTrPerfDetails.head());

lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution

lEngine.standrdPlots("outputs/sof_example");

dfapp_in = df.copy();
dfapp_in.tail()

#H = 12
dfapp_out = lEngine.forecast(dfapp_in, H);
dfapp_out.to_csv("outputs/sof_example_apply_out.csv")
dfapp_out.tail(2 * H)
print("Forecast Columns " , dfapp_out.columns);
Forecast_DF = dfapp_out[[lDateVar , lSignalVar,
                         lSignalVar + '_Forecast' ,
                         lSignalVar + '_Forecast_Lower_Bound',
                         lSignalVar + '_Forecast_Upper_Bound']]
print(Forecast_DF.info())
print("Forecasts\n" , Forecast_DF.tail(2*H));

print("\n\n<ModelInfo>")
print(lEngine.to_json());
print("</ModelInfo>\n\n")
print("\n\n<Forecast>")
print(Forecast_DF.to_json(date_format='iso'))
print("</Forecast>\n\n")

