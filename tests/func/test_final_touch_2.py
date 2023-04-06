import numpy as np
import pandas as pd

if __name__ == '__main__':
    N = 16
    np.random.seed(seed=1960)
    df_train = pd.DataFrame({"Date" : np.arange(N),
                             "Signal" : [2*x + 1 for x in np.arange(N)]})
    print(df_train.head(N))
    
    import pyaf.ForecastEngine as autof
    lEngine = autof.cForecastEngine()
    lEngine.mOptions.set_active_transformations(['None'])
    lEngine.mOptions.set_active_trends(['LinearTrend'])

    
    lEngine.train(iInputDS = df_train, iTime = 'Date', iSignal = 'Signal', iHorizon = 2);
    lEngine.getModelInfo()
    lPerfs = lEngine.mSignalDecomposition.mTrPerfDetailsBySignal['Signal']
    print(lPerfs.columns)
    print(lPerfs[['Model' , 'Forecast_MASE_1', 'Forecast_MASE_2']].values);

    lEngine.standardPlots("outputs/FT_2_sixteen_rows_")
    
    df_forecast = lEngine.forecast(iInputDS = df_train, iHorizon = 2)
    print(df_forecast.columns) #
    for (idx, col) in enumerate(df_forecast.columns):
        print("CHECK_COLUMN_DATA", idx, col, df_forecast[col].dtype, df_forecast[col].values.round(3).tolist())
