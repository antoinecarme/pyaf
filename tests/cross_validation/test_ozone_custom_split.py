from __future__ import absolute_import

import pandas as pd
import numpy as np


import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds



def test_ozone_custom(folds, start_validation_at_fold):
    b1 = tsds.load_ozone()
    df = b1.mPastData

    lEngine = autof.cForecastEngine()
    lEngine

    H = b1.mHorizon;
    lRatio = 1.0 / folds
    lEngine.mOptions.mCustomSplit = (lRatio * start_validation_at_fold, lRatio, 0.0);
    lEngine.mOptions.mDebugPerformance = True;
    lEngine.train(df , b1.mTimeVar , b1.mSignalVar, H);
    lEngine.getModelInfo();
    

    print(lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution)

    lEngine.standardPlots("outputs/my_ozone_custom_split_" + str(folds) + "_" + str(start_validation_at_fold));

    dfapp_in = df.copy();
    dfapp_in.tail()

    #H = 12
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



nfolds = 5
for i in range(nfolds - 1):
    test_ozone_custom(nfolds , i + 1)
