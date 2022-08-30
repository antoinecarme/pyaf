import numpy as np
import pandas as pd

if __name__ == '__main__':
    # generate a daily signal covering one year 2016 in a pandas dataframe
    N = 365
    np.random.seed(seed=1960)
    df_train = pd.DataFrame({"Date" : pd.date_range(start="2016-01-25", periods=N, freq='H'),
                             "Signal" : (np.arange(N)//40 + np.arange(N) % 21 + np.random.randn(N))})
    # print(df_train.head(N))
    
    import pyaf.ForecastEngine as autof
    # create a forecast engine. This is the main object handling all the operations
    lEngine = autof.cForecastEngine()
    
    # get the best time series model for predicting one week
    lEngine.train(iInputDS = df_train, iTime = 'Date', iSignal = 'Signal', iHorizon = 7);
    lEngine.getModelInfo() # => relative error 7% (MAPE)

    lEngine.standardPlots("outputs/bug_hourly_");
    

