# Thanks to https://colab.research.google.com/drive/1zaVQuobR8M63qB-UDDX8ZX37ctl98YIT
# Marian W. : Predykcje niestety dość mocno mijają się z danymi historycznymi.
# Translation from polish : Predictions, unfortunately, are quite far from historical data.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def prepare_dataset():
    df = pd.read_csv("data/real-life/veturilo.csv", parse_dates=True, usecols=["ts","qnty"], index_col="ts" )
    df.qnty.unique()
    df.qnty = df.qnty.replace("?", np.NaN).fillna(method='ffill').astype('uint8')
    df = df.qnty.resample("H").mean().to_frame()
    
    df_predict = df.iloc[:]

    df_tmp = df_predict.reset_index()
    df_tmp.columns = ['ds','y']
    df_tmp.index = (0,) * len(df_tmp)
    df_tmp.index.name = 'unique_id'
    return df_tmp

import pyaf.ForecastEngine as autof


df = prepare_dataset()
horizon = 48

Y_train_df = df[:-horizon]
Y_test_df = df[-horizon:]

lEngine = autof.cForecastEngine()
lEngine.mOptions.mModelSelection_Criterion = "RMSE"
lEngine.mOptions.mCycle_Criterion = "RMSE"
lEngine.train(iInputDS = Y_train_df, iTime = 'ds', iSignal = 'y', iHorizon = horizon)

lEngine.getModelInfo()

print("\n\n<ModelInfo>")
print(lEngine.to_json());
print("</ModelInfo>\n\n")

forecast_df= lEngine.forecast(Y_train_df, horizon)

print(forecast_df.head(horizon))
print(forecast_df.tail(horizon))

lEngine.standardPlots("outputs/veturilo_RMSE")
