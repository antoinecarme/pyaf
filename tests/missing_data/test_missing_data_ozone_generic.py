
import pandas as pd
import numpy as np


import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds

def add_some_missing_data_in_signal(df, col):
    lRate = 0.2
    df.loc[df.sample(frac=lRate, random_state=1960).index, col] = np.nan
    return df

def add_some_missing_data_in_time(df, col):
    lRate = 0.2
    df.loc[df.sample(frac=lRate, random_state=1960).index, col] = np.nan
    return df


def test_ozone_missing_data(iTimeMissingDataImputation, iSignalMissingDataImputation):

    b1 = tsds.load_ozone()
    df = b1.mPastData

    if(iTimeMissingDataImputation is not None):
        df = add_some_missing_data_in_time(df, b1.mTimeVar)
    if(iSignalMissingDataImputation is not None):
        df = add_some_missing_data_in_signal(df, b1.mSignalVar)
        
    lEngine = autof.cForecastEngine()
    H = b1.mHorizon;
    lEngine.mOptions.mMissingDataOptions.mTimeMissingDataImputation = iTimeMissingDataImputation
    lEngine.mOptions.mMissingDataOptions.mSignalMissingDataImputation = iSignalMissingDataImputation
    lEngine.train(df , b1.mTimeVar , b1.mSignalVar, H);
    lEngine.getModelInfo();

    lEngine.standardPlots("outputs/ozone_missing_data_" + str(iTimeMissingDataImputation) + "_" + str(iSignalMissingDataImputation))

    dfapp_in = df.copy();
    dfapp_in.tail()
    dfapp_out = lEngine.forecast(dfapp_in, H);
    #dfapp_out.to_csv("outputs/ozone_apply_out.csv")
    dfapp_out.tail(2 * H)
    print("Forecast Columns " , dfapp_out.columns);
    Forecast_DF = dfapp_out[[b1.mTimeVar , b1.mSignalVar, b1.mSignalVar + '_Forecast']]
    print(Forecast_DF.info())
    print("Forecasts\n" , Forecast_DF.tail(H));

    print("\n\n<ModelInfo>")
    print(lEngine.to_json());
    print("</ModelInfo>\n\n")
    print("\n\n<Forecast>")
    print(Forecast_DF.tail(2*H).to_json(date_format='iso'))
    print("</Forecast>\n\n")

