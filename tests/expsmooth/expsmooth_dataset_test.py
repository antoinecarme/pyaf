import pandas as pd
import numpy as np
import pyaf.ForecastEngine as autof

import logging
import logging.config

logger = logging.getLogger()
logger.handlers = []

logging.basicConfig(level=logging.INFO)


def analyze_dataset(name , horizon):
    signal = name.replace(".csv" , "") 
    url="https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/R_TSData/expsmooth/" + name 
    df = pd.read_csv(url)
    df = df [[df.columns[0] , df.columns[-1]]].dropna()
    df.columns = ['Date' , signal]
    lEngine = autof.cForecastEngine()
    # lEngine.mOptions.enable_slow_mode();
    N = df.shape[0]
    lEngine.train(df , 'Date' , signal , horizon)
    lEngine.getModelInfo()
    lEngine.standardPlots("outputs/expsmooth_" + name + "_" + str(horizon))
    lPerf_H = lEngine.mSignalDecomposition.mBestModel.mForecastPerfs[signal + "_Forecast_" + str(horizon)]
    print("PERFORMANCE MAPE_FORECAST" , signal, lPerf_H.mMAPE)
    return lEngine

