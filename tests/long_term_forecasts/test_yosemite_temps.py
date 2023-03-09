import pandas as pd
import numpy as np
import pyaf.ForecastEngine as autof
import sys, datetime

def buildModel(H):
    # thanks to facebook/prophet.
    # Temperature every 5 minutes from 2017-05-01 00:00:00 to 2017-07-05 00:00:00
    # 18722 observations
    # Missing data around : 11689 2017-06-10 14:05:00          NaN

    uri = "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_yosemite_temps.csv"
    lTimeVar, lSignalVar = "Time", "Temperature"
    df = pd.read_csv(uri)
    df.columns = [lTimeVar, lSignalVar]
    df[lTimeVar] = df[lTimeVar].apply(datetime.datetime.fromisoformat)
    N = df.shape[0];
    print(df.head())
    print(df[df.Temperature.isnull()].head())
    print(df.info())
    print(df.describe())
    sys.stdout.flush()

    K = 1000
    df_train = df[0:N-K]

    lEngine = autof.cForecastEngine()
    lEngine

    # lEngine.mOptions.enable_slow_mode();
    lEngine.mOptions.mDebug = True;
    lEngine.mOptions.mDebugPerformance = True;
    # lEngine.mOptions.mParallelMode = False
    lEngine.mOptions.mMissingDataOptions.mSignalMissingDataImputation = "PreviousValue"
    lEngine.mOptions.mActiveTrends['Lag1Trend'] = False
    
    lEngine.train(df_train , lTimeVar , lSignalVar, H);
    lEngine.getModelInfo();
    
    
    lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution
    
    lEngine.standardPlots("outputs/my_yosemite_series_" + str(H));

    dfapp_in = df_train.copy();
    dfapp_in.tail()
    
    #H = 12
    dfapp_out = lEngine.forecast(dfapp_in, H);
    #dfapp_out.to_csv("outputs/ozone_apply_out.csv")
    dfapp_out.tail(2 * H)
    print("Forecast Columns " , dfapp_out.columns);
    Forecast_DF = dfapp_out[[lTimeVar , lSignalVar, lSignalVar + '_Forecast', lSignalVar + '_Forecast_Lower_Bound', lSignalVar + '_Forecast_Upper_Bound']]
    print(Forecast_DF.info())
    print("Forecasts\n" , Forecast_DF.tail(H));

    print("\n\n<ModelInfo>")
    print(lEngine.to_json());
    print("</ModelInfo>\n\n")
    print("\n\n<Forecast>")
    print(Forecast_DF.tail(2*H).to_json(date_format='iso'))
    print("</Forecast>\n\n")

