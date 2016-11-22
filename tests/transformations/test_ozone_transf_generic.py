import pandas as pd
import numpy as np
import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds


def test_transformation(itransformation):

    b1 = tsds.load_ozone()
    df = b1.mPastData

    lEngine = autof.cForecastEngine()
    lEngine

    H = b1.mHorizon;
    # lEngine.mOptions.enable_slow_mode();
    lEngine.mOptions.mDebugPerformance = True;

    lEngine.mOptions.disable_all_transformations();
    lEngine.mOptions.mActiveTransformation[itransformation] = True;
    lEngine.mOptions.mBoxCoxOrders = lEngine.mOptions.mExtensiveBoxCoxOrders;

    lEngine.train(df , b1.mTimeVar , b1.mSignalVar, H);
    lEngine.getModelInfo();
    print(lEngine.mSignalDecomposition.mTrPerfDetails.head());

    lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution
    
    lEngine.standrdPlots("outputs/my_ozone_" + itransformation);
    
    dfapp_in = df.copy();
    dfapp_in.tail()
    
    dfapp_out = lEngine.forecast(dfapp_in, H);
    dfapp_out.to_csv("outputs/ozone_apply_out_" + itransformation + ".csv")
    dfapp_out.tail(H)
    print("Forecast Columns " , dfapp_out.columns);
    Forecast_DF = dfapp_out[[b1.mTimeVar , b1.mSignalVar, b1.mSignalVar + '_Forecast']]
    print(Forecast_DF.info())
    print("Forecasts\n" , Forecast_DF.tail(H));

    print("\n\n<ModelInfo>")
    print(lEngine.to_json());
    print("</ModelInfo>\n\n")
    print("\n\n<Forecast>")
    print(Forecast_DF.head(H).to_json(date_format='iso'))
    print("</Forecast>\n\n")
