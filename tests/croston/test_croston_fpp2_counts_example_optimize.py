import pandas as pd

def create_dataset():
    lCounts = "0 2 0 1 0 11 0 0 0 0 2 0 6 3 0 0 0 0 0 7 0 0 0 0 0 0 0 3 1 0 0 1 0 1 0 0".split()
    lCounts = [float(c) for c in lCounts]
    N = len(lCounts)
    lDates = pd.date_range(start="2000-01-01", periods=N, freq='m')

    df = pd.DataFrame({"Date" : lDates, "Signal" : lCounts})
    return df

def create_model(croston_type):
    df_train = create_dataset()
    # print(df_train.head(N))

    import pyaf.ForecastEngine as autof
    lEngine = autof.cForecastEngine()

    lEngine.mOptions.set_active_trends(['ConstantTrend', 'LinearTrend'])
    lEngine.mOptions.set_active_periodics(['NoCycle'])
    lEngine.mOptions.set_active_transformations(['None'])
    lEngine.mOptions.set_active_autoregressions(['CROSTON'])
    lEngine.mOptions.mModelSelection_Criterion = "L2";
    lEngine.mOptions.mCrostonOptions.mMethod = croston_type
    lEngine.mOptions.mCrostonOptions.mZeroRate = 0.0
    lEngine.mOptions.mCrostonOptions.mAlpha = None
    
    # get the best time series model for predicting one week
    lEngine.train(iInputDS = df_train, iTime = 'Date', iSignal = 'Signal', iHorizon = 7);

    lEngine.getModelInfo()

    lName = "outputs/fpp2_croston_optimize_" + str(croston_type) + "_"
    lEngine.standardPlots(lName);

    # predict one week
    df_forecast = lEngine.forecast(iInputDS = df_train, iHorizon = 7)
    # list the columns of the forecast dataset
    print(df_forecast.columns) #

    cols = ['Date', 'Signal', '_Signal',
            '_Signal_TransformedForecast', 'Signal_Forecast']
    # print the real forecasts
    print(df_forecast[cols].tail(12))
    
    print(df_forecast['Signal'].describe())
    print(df_forecast['Signal_Forecast'].describe())


create_model(None)
