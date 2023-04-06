import numpy as np
import pandas as pd

def create_training_dataset(iSigType = None):
    # generate a daily signal covering one year 2016 in a pandas dataframe
    N = 365
    np.random.seed(seed=1960)
    df_train = pd.DataFrame({"Date" : pd.date_range(start="2016-01-25", periods=N, freq='D'),
                             "Signal" : (np.arange(N) + np.arange(N) % 21 + np.random.randn(N))})
    # print(df_train.head(N))
    df_train["Signal"] = df_train["Signal"] - df_train["Signal"].mean()
    if(iSigType == 'Neg'):
        df_train["Signal"] = -df_train["Signal"]
    if(iSigType == 'AddConst'):
        df_train["Signal"] = 100 + df_train["Signal"]
    if(iSigType == 'MulConst'):
        df_train["Signal"] = 10 * df_train["Signal"]
    return df_train

def display_forecast_columns(df_forecast, iEngine):
    # list the columns of the forecast dataset
    print(df_forecast.columns) #
    lColumns = ['Date', 'Signal', 'row_number', 'Date_Normalized', 'Signal_Transformed']
    print(df_forecast[lColumns].head(5)) #
    print(df_forecast[lColumns].tail(5)) 

    lPrefix = iEngine.mSignalDecomposition.mBestModel.mSignal
    lColumns2 = [lPrefix + '_Trend', lPrefix + '_Trend_residue']
    print(df_forecast[lColumns2].head(5)) #
    print(df_forecast[lColumns2].tail(5)) 
    lColumns2 = [lPrefix + '_Cycle', lPrefix + '_Cycle_residue']
    print(df_forecast[lColumns2].head(5)) #
    print(df_forecast[lColumns2].tail(5)) 
    lColumns2 = [lPrefix + '_AR', lPrefix + '_AR_residue']
    print(df_forecast[lColumns2].head(5)) #
    print(df_forecast[lColumns2].tail(5)) 
    lColumns2 = [lPrefix + '_TransformedForecast', lPrefix + '_TransformedResidue']
    print(df_forecast[lColumns2].head(5)) #
    print(df_forecast[lColumns2].tail(5)) 
    lColumns2 = ['Signal_Forecast', 'Signal_Residue']
    print(df_forecast[lColumns2].head(5)) #
    print(df_forecast[lColumns2].tail(5)) 

    # print the real forecasts
    # Future dates : ['2017-01-19T00:00:00.000000000' '2017-01-20T00:00:00.000000000' '2017-01-21T00:00:00.000000000' '2017-01-22T00:00:00.000000000' '2017-01-23T00:00:00.000000000' '2017-01-24T00:00:00.000000000' '2017-01-25T00:00:00.000000000']
    print(df_forecast['Date'].tail(7).values)

    # signal forecast : [ 9.74934646  10.04419761  12.15136455  12.20369717  14.09607727 15.68086323  16.22296559]
    print(df_forecast['Signal_Forecast'].tail(7).values)
    

def test_transform(iTranform, df_train):
    # generate a daily signal covering one year 2016 in a pandas dataframe
    import pyaf.ForecastEngine as autof
    # create a forecast engine. This is the main object handling all the operations
    lEngine = autof.cForecastEngine()
    lEngine.mOptions.set_active_transformations([ iTranform ])
    lEngine.mOptions.set_active_trends(['ConstantTrend'])

    # get the best time series model for predicting one week
    lEngine.train(iInputDS = df_train, iTime = 'Date', iSignal = 'Signal', iHorizon = 7);
    lEngine.getModelInfo() # => relative error 7% (MAPE)

    lEngine.standardPlots("transform_stability_test_" + iTranform)
    
    # predict one week
    df_forecast = lEngine.forecast(iInputDS = df_train, iHorizon = 7)

    display_forecast_columns(df_forecast, lEngine)


def test_all_trsnaformations(iSigType = None):

    lKnownTransformations = ['None', 'Difference', 'RelativeDifference',
                             'Integration', 'BoxCox',
                             'Quantization', 'Logit',
                             'Fisher', 'Anscombe'];

    df_train = create_training_dataset(iSigType)
    for lTransform in lKnownTransformations:
        test_transform(lTransform , df_train)
