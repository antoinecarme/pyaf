import numpy as np
import pandas as pd

if __name__ == '__main__':
    # generate a daily signal covering one year 2016 in a pandas dataframe
    N = 365
    np.random.seed(seed=1960)
    df_train = pd.DataFrame({"Date" : pd.date_range(start="2016-01-25", periods=N, freq='D'),
                             "Signal" : (np.arange(N)//40 + np.arange(N) % 21 + np.random.randn(N))})
    # print(df_train.head(N))
    
    import pyaf.ForecastEngine as autof
    # create a forecast engine. This is the main object handling all the operations
    lEngine = autof.cForecastEngine()
    
    # get the best time series model for predicting one week
    lEngine.train(iInputDS = df_train, iTime = 'Date', iSignal = 'Signal', iHorizon = 7);
    lEngine.getModelInfo() # => relative error 7% (MAPE)

    # predict one week
    df_forecast = lEngine.forecast(iInputDS = df_train, iHorizon = 7)
    # list the columns of the forecast dataset
    print(df_forecast.columns) #

    # print the real forecasts
    # Future dates : ['2017-01-19T00:00:00.000000000' '2017-01-20T00:00:00.000000000' '2017-01-21T00:00:00.000000000' '2017-01-22T00:00:00.000000000' '2017-01-23T00:00:00.000000000' '2017-01-24T00:00:00.000000000' '2017-01-25T00:00:00.000000000']
    print(df_forecast['Date'].tail(7).values)

    # signal forecast : [ 9.74934646  10.04419761  12.15136455  12.20369717  14.09607727 15.68086323  16.22296559]
    print(df_forecast['Signal_Forecast'].tail(7).values)
