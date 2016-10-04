import pandas as pd
import numpy as np
import AutoForecast.ForecastEngine as autof
import AutoForecast.Bench.TS_datasets as tsds

#get_ipython().magic('matplotlib inline')

b1 = tsds.load_ozone()
df = b1.mPastData


for k in range(1,32):
    df[b1.mTimeVar + "_" + str(k) + '_Daily'] = pd.date_range('2000-1-1', periods=df.shape[0], freq=str(k) + 'D')


df.to_csv("outputs/ozone_WDHMS.csv");
#df.tail(10)
#df[:-10].tail()
#df[:-10:-1]
#df.describe()


for k in range(1,32):
    for timevar in [b1.mTimeVar +  "_" + str(k) + '_Daily']:

        lEngine = autof.cForecastEngine()
        lEngine

        H = b1.mHorizon;
        # lEngine.mOptions.enable_slow_mode();
        lEngine.mOptions.mDebugPerformance = True;
        lEngine.train(df , timevar , b1.mSignalVar, H);
        lEngine.getModelInfo();
        print(lEngine.mSignalDecomposition.mTrPerfDetails.head());

        lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution
        
        dfapp_in = df.copy();
        dfapp_in.tail()
        
        # H = 12
        dfapp_out = lEngine.forecast(dfapp_in, H);
        dfapp_out.to_csv("outputs/ozone_" + timevar + "apply_out.csv")
        dfapp_out.tail(2 * H)
        print("Forecast Columns " , dfapp_out.columns);
        Forecast_DF = dfapp_out[[timevar , b1.mSignalVar, b1.mSignalVar + '_Forecast']]
        print(Forecast_DF.info())
        print("Forecasts\n" , Forecast_DF.tail(H));

        print("\n\n<ModelInfo>")
        print(lEngine.to_json());
        print("</ModelInfo>\n\n")
        print("\n\n<Forecast>")
        print(Forecast_DF.to_json(date_format='iso'))
        print("</Forecast>\n\n")
        
        lEngine.standrdPlots(name = "outputs/ozone_" + timevar)
