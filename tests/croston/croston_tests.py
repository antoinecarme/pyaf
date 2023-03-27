import numpy as np
import pandas as pd

def create_intermittent_signal(N):
    np.random.seed(seed=1960)
    sig = np.zeros(N)
    for i in range(0, N // 30):
        if(np.random.random() < 0.5):
            sig[i * 30] = np.random.randint(100)
    return sig



# create an intemittent signal with a linear trend
def create_intermittent_signal_linear_trend(N):
    np.random.seed(seed=1960)
    sig = [k/ N for k in range(N)]
    for i in range(0, N // 30):
        if(np.random.random() < 0.5):
            sig[i * 30] = sig[i * 30] + 0.5 * np.random.random()
    return sig

def create_model(N = 365 , croston_type=None, iTrend = False):
    # N = 365
    np.random.seed(seed=1960)
    signal = None
    if(iTrend):
        signal = create_intermittent_signal_linear_trend(N)
    else:
        signal = create_intermittent_signal(N)
        
    df_train = pd.DataFrame({"Date" : pd.date_range(start="2016-01-25", periods=N, freq='D'),
                             "Signal" : signal})
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
    
    # get the best time series model for predicting one week
    lEngine.train(iInputDS = df_train, iTime = 'Date', iSignal = 'Signal', iHorizon = 7);

    lEngine.getModelInfo()

    lName = "outputs/croston_" + str(croston_type) + "_"
    lName = lName + ("linear_trend" if iTrend else "no_trend" )
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
