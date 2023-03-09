import pandas as pd
import numpy as np
import pyaf.ForecastEngine as autof

np.random.seed(seed=1960)

#get_ipython().magic('matplotlib inline')

df = pd.DataFrame()

lTimeVar = 'Time'
lSignalVar = 'Signal'

N = 10000
df[lTimeVar + '_Hourly'] = pd.date_range('2000-1-1', periods=N, freq='1h')

df['Hour'] =  df[lTimeVar + '_Hourly'].dt.hour
df['Day'] =  df[lTimeVar + '_Hourly'].dt.dayofweek


df[lSignalVar] = 5 + np.random.randn(N) +  10 * df['Hour'].apply(lambda x : x if (12 <= x and x < 14) else 23) *  df['Day'].apply(lambda x : x if (x < 4) else 12)  

print(df.head())
print(df.info())



#df.to_csv("outputs/ozone_WDHMS.csv");
#df.tail(10)
#df[:-10].tail()
#df[:-10:-1]
#df.describe()


for k in [1]:
    for timevar in [lTimeVar +  '_Hourly']:

        lEngine = autof.cForecastEngine()
        lEngine

        H = 24;
        # lEngine.mOptions.enable_slow_mode();
        lEngine.mOptions.mDebugPerformance = True;
        lEngine.mOptions.mFilterSeasonals = True;
        lEngine.mOptions.mDebugCycles = False;
        lEngine.mOptions.set_active_autoregressions([]);

        lEngine.train(df , timevar , lSignalVar, H);
        lEngine.getModelInfo();
        

        lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution
        
        dfapp_in = df.copy();
        dfapp_in.tail()
        
        # H = 12
        dfapp_out = lEngine.forecast(dfapp_in, H);
        #dfapp_out.to_csv("outputs/ozone_" + timevar + "apply_out.csv")
        dfapp_out.tail(2 * H)
        print("Forecast Columns " , dfapp_out.columns);
        Forecast_DF = dfapp_out[[timevar , lSignalVar, lSignalVar + '_Forecast']]
        print(Forecast_DF.info())
        print("Forecasts\n" , Forecast_DF.tail(H));

        print("\n\n<ModelInfo>")
        print(lEngine.to_json());
        print("</ModelInfo>\n\n")
        print("\n\n<Forecast>")
        print(Forecast_DF.tail(2*H).to_json(date_format='iso'))
        print("</Forecast>\n\n")
        
        lEngine.standardPlots(name = "outputs/ozone_LunchTime_" + timevar)
