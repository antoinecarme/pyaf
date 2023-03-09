import pandas as pd
import numpy as np
import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds

#get_ipython().magic('matplotlib inline')

import datetime as dt
from dateutil.relativedelta import relativedelta

def addMinutes(iTime , iMinutes):
    lTime = iTime
    date_after_minutes = lTime + relativedelta(minutes=iMinutes)
    return np.datetime64(date_after_minutes) 

def replicate(df , K):
    N = df.shape[0];
    y = pd.Series(range(K*N));
    tmin = df.Time.min()
    Time1 = y.apply(lambda x : addMinutes(tmin , x))
    # Time1.describe()
    df1 = pd.DataFrame();
    df1['Time'] = Time1.values;
    df1['Ozone'] = list(df['Ozone'].values)*K;
    print(df1['Ozone'].describe());
    return df1;

b1 = tsds.load_ozone()
df = b1.mPastData

NH = 1000
df1 = replicate(df, NH);
df1.head()

#df.tail(10)
#df[:-10].tail()
#df[:-10:-1]
#df.describe()


lEngine = autof.cForecastEngine()
lEngine

H = NH // 4;
# lEngine.mOptions.enable_slow_mode();
lEngine.mOptions.mDebug = True;
lEngine.mOptions.mDebugPerformance = True;
lEngine.mOptions.mParallelMode = False 

lEngine.train(df1 , b1.mTimeVar , b1.mSignalVar, H);
lEngine.getModelInfo();


lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution

# lEngine.standardPlots("outputs/my_long_ozone_series_");

dfapp_in = df1.copy();
dfapp_in.tail()

#H = 12
dfapp_out = lEngine.forecast(dfapp_in, H);
#dfapp_out.to_csv("outputs/ozone_apply_out.csv")
dfapp_out.tail(2 * H)
print("Forecast Columns " , dfapp_out.columns);
Forecast_DF = dfapp_out[[b1.mTimeVar , b1.mSignalVar, b1.mSignalVar + '_Forecast']]
print(Forecast_DF.info())
print("Forecasts\n" , Forecast_DF.tail(H));

print("\n\n<ModelInfo>")
print(lEngine.to_json());
print("</ModelInfo>\n\n")
print("\n\n<Forecast>")
print(Forecast_DF.tail(2*H).to_json(date_format='iso'))
print("</Forecast>\n\n")

