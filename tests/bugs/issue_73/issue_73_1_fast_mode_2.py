import numpy as np
import pandas as pd


df_train = pd.read_csv("https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/R_TSData/taylor.csv")

import pyaf.ForecastEngine as autof

lEngine = autof.cForecastEngine()
lEngine.mOptions.mCycleLengths = None

lEngine.train(iInputDS = df_train, iTime = 'time', iSignal = 'signal', iHorizon = 36);
lEngine.getModelInfo() #

lEngine.standardPlots("outputs/issue_73_1_fast_mode_more_cycles");
