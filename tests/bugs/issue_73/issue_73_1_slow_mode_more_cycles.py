import numpy as np
import pandas as pd


df_train = pd.read_csv("https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/R_TSData/taylor.csv")

import pyaf.ForecastEngine as autof

lEngine = autof.cForecastEngine()
lEngine.mOptions.enable_slow_mode()
lEngine.mOptions.mCycleLengths = range(2,500);

lEngine.train(iInputDS = df_train, iTime = 'time', iSignal = 'signal', iHorizon = 36);
lEngine.getModelInfo() #

lEngine.standardPlots("outputs/issue_73_1_slow_mode_more_cycles");
