import pandas as pd
import pyaf.ForecastEngine as autof

df = pd.read_csv("https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/fpp2/insurance.csv")


(lTimeVar , lSigVar , lExogVar) = ("Index", "Quotes" , "TV.advert")
df_sig = df[[lTimeVar , lSigVar]]
df_exog = df[[lTimeVar , lExogVar]] # need time here
H = 4

lEngine = autof.cForecastEngine()
lEngine.mOptions.set_active_autoregressions(['ARX'])

lExogenousData = (df_exog , [lExogVar]) 
lEngine.train(df_sig , lTimeVar , lSigVar, H, lExogenousData);

lEngine.getModelInfo();

lEngine.standardPlots("outputs/fpp2_insurance_submitted")
