import numpy as np
import pandas as pd

df = pd.read_csv('tests/bugs/issue_19/issue_19_data_1.csv')

import datetime

def convert_date(x):
    y = np.nan
    try:
        y = datetime.datetime.strptime(str(x), "%Y")
    except:
        # bad format
        pass
    return y


df['date'] = df['date'].apply(convert_date)

df_train = df[['date' , 'number']].dropna().reset_index(drop=True)

print(df_train)

import pyaf.ForecastEngine as autof
lEngine = autof.cForecastEngine()

lEngine.train(iInputDS = df_train, iTime = 'date', iSignal = 'number', iHorizon = 7);
print(lEngine.getModelInfo())


# lEngine.standardPlots('outputs/tour')

df_forecast = lEngine.forecast(iInputDS = df_train, iHorizon = 7)
print(df_forecast.columns)
print(df_forecast[['date', 'number_Forecast', 'number_Forecast_Lower_Bound', 'number_Forecast_Upper_Bound']].tail(7))
