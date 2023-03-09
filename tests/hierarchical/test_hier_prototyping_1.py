import pandas as pd
import numpy as np
import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds

import datetime

#get_ipython().magic('matplotlib inline')


trainfile = "data/Hierarchical/hts_dataset.csv"
lDateColumn = 'Date'

df = pd.read_csv(trainfile, sep=r',', engine='python', skiprows=0);
df[lDateColumn] = df[lDateColumn].apply(lambda x : datetime.datetime.strptime(x, "%Y-%m-%d"))

print(df.tail(10))
#df[:-10].tail()
#df[:-10:-1]
print(df.info())
print(df.describe())

lBottomColumns = [col for col in df.columns if col != lDateColumn]

print("BOTTOM_COLUMNS" , lBottomColumns);

H = 4;

for signal in lBottomColumns:
    lEngine = autof.cForecastEngine()
    lEngine

    # lEngine.mOptions.enable_slow_mode();
    # lEngine.mOptions.mDebugPerformance = True;
    lEngine.mOptions.set_active_autoregressions([]);
    lEngine.train(df , lDateColumn , signal, H);
    lEngine.getModelInfo();
    

    lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution

    #lEngine.standardPlots("outputs/hierarchical_" + signal);

    dfapp_in = df.copy();
    dfapp_in.tail()

    dfapp_out = lEngine.forecast(dfapp_in, H);
    #dfapp_out.to_csv("outputs/ozone_apply_out.csv")
    dfapp_out.tail(2 * H)
    print("Forecast Columns " , dfapp_out.columns);
    Forecast_DF = dfapp_out[[lDateColumn , signal, signal + '_Forecast']]
    print(Forecast_DF.info())
    print("Forecasts\n" , Forecast_DF.tail(H));

    print("\n\n<ModelInfo>")
    print(lEngine.to_json());
    print("</ModelInfo>\n\n")
    print("\n\n<Forecast>")
    print(Forecast_DF.tail(2*H).to_json(date_format='iso'))
    print("</Forecast>\n\n")

