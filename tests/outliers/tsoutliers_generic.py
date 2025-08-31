
import pandas as pd
import numpy as np


import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds


def process_tsoutliers_signal(csv_file, iconv_time = True):
    df = pd.read_csv(csv_file)
    lTimeVar , lSignalVar = df.columns
    if(iconv_time):
        df[lTimeVar] = pd.to_datetime(df[lTimeVar])
    lEngine = autof.cForecastEngine()
    lEngine
    lEngine.mOptions.mMissingDataOptions.mSignalMissingDataImputation = "PreviousValue"
    
    H = 12 # months
    lEngine.train(df , lTimeVar , lSignalVar, H);
    lEngine.getModelInfo();


    lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution

    lEngine.standardPlots("outputs/my_tsoutliers_" + lSignalVar + "_");

    dfapp_in = df.copy();
    dfapp_in.tail()

    dfapp_out = lEngine.forecast(dfapp_in, H);
    dfapp_out.tail(2 * H)
    print("Forecast Columns " , dfapp_out.columns);
    Forecast_DF = dfapp_out[[lTimeVar , lSignalVar, lSignalVar + '_Forecast']]
    print(Forecast_DF.info())
    print("Forecasts\n" , Forecast_DF.tail(H));

    print("\n\n<ModelInfo>")
    print(lEngine.to_json());
    print("</ModelInfo>\n\n")
    print("\n\n<Forecast>")
    print(Forecast_DF.tail(2*H).to_json(date_format='iso'))
    print("</Forecast>\n\n")


# process_tsoutliers_signal("https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/outliers/cran_tsoutliers_bde9915_euprin.csv")
