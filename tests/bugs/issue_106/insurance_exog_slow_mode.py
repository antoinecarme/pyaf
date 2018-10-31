import numpy as np
import pandas as pd
import pyaf.ForecastEngine as autof

# example from https://otexts.org/fpp2/lagged-predictors.html
df = pd.read_csv("https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/fpp2/insurance.csv")

df.info()

(lTimeVar , lSigVar , lExogVar) = ("Index", "Quotes" , "TV.advert")
df_sig = df[[lTimeVar , lSigVar]]
df_exog = df[[lTimeVar , lExogVar]] # need time here
H = 4

df_sig.info()
df_exog.info()

lEngine = autof.cForecastEngine()
lEngine.mOptions.enable_slow_mode()
lEngine.mOptions.mDebug = True;
lEngine.mOptions.mDebugCycles = True;
lEngine.mOptions.mDebugProfile = True;
lEngine.mOptions.mDebugPerformance = True;


lExogenousData = (df_exog , [lExogVar]) 
lEngine.train(df_sig , lTimeVar , lSigVar, H, lExogenousData);

lEngine.getModelInfo();
lEngine.standardPlots(name = "outputs/insurance_slow")

dfapp_in = df_sig.copy();
dfapp_in.tail()

dfapp_out = lEngine.forecast(dfapp_in, H);
dfapp_out.tail(2 * H)
print("Forecast Columns " , dfapp_out.columns);
Forecast_DF = dfapp_out[[lTimeVar , lSigVar, lSigVar + '_Forecast']]
print(Forecast_DF.info())
print("Forecasts\n" , Forecast_DF.tail(H).values);
    
print("\n\n<ModelInfo>")
print(lEngine.to_json());
print("</ModelInfo>\n\n")
print("\n\n<Forecast>")
print(Forecast_DF.tail(2*H).to_json(date_format='iso'))
print("</Forecast>\n\n")

