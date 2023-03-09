import pandas as pd
import numpy as np
import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds

#get_ipython().magic('matplotlib inline')

b1 = tsds.load_ozone()
df = b1.mPastData


df[b1.mTimeVar + '_BusinessDaily'] = pd.bdate_range('2000-1-1', periods=df.shape[0])


#df.to_csv("outputs/ozone_WDHMS.csv");
#df.tail(10)
#df[:-10].tail()
#df[:-10:-1]
#df.describe()


timevar = b1.mTimeVar +  '_BusinessDaily'

lEngine = autof.cForecastEngine()
lEngine

H = b1.mHorizon;
# lEngine.mOptions.enable_slow_mode();
# lEngine.mOptions.mDebugPerformance = True;
lEngine.mOptions.set_active_autoregressions([]);

lEngine.train(df , timevar , b1.mSignalVar, H);
lEngine.getModelInfo();


lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution

lEngine.standardPlots("outputs/my_ozone_bday");


dfapp_in = df.copy();
dfapp_in.tail()
        
# H = 12
dfapp_out = lEngine.forecast(dfapp_in, H);
#dfapp_out.to_csv("outputs/ozone_" + timevar + "apply_out.csv")
dfapp_out.tail(2 * H)
print("Forecast Columns " , dfapp_out.columns);
Forecast_DF = dfapp_out[[timevar , b1.mSignalVar, b1.mSignalVar + '_Forecast']]
print(Forecast_DF.info())
print("Forecasts\n" , Forecast_DF.tail(2*H));

print("\n\n<ModelInfo>")
print(lEngine.to_json());
print("</ModelInfo>\n\n")
print("\n\n<Forecast>")
print(Forecast_DF.tail(2*H).to_json(date_format='iso'))
print("</Forecast>\n\n")
        
# lEngine.standardPlots(name = "outputs/ozone_" + timevar)
