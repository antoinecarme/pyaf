import pandas as pd
import numpy as np

import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds


def build_model(model, backends):
    print("BUILD_PYTORCH_OR_KERAS_MODEL", (model, backends))
    b1 = tsds.load_ozone_exogenous()
    df = b1.mPastData
    H = b1.mHorizon;
    lBackEnds = "_".join([x for x in backends])

    df1 = df
    lEngine = autof.cForecastEngine()
    lSignal = b1.mSignalVar + "_" + model + "_" + lBackEnds
    df[lSignal] = df[b1.mSignalVar]
    lEngine
    #    lEngine.mOptions.mEnableSeasonals = False;
    #    lEngine.mOptions.mEnableCycles = False;
    #    lEngine.mOptions.mEnableARModels = False;
    #    lEngine.mOptions.mDebugCycles = False;
    # lEngine.mOptions.mParallelMode = False;
    lEngine.mOptions.set_active_autoregressions([model]);
    lEngine.mOptions.mDL_Backends = backends;
    
    lExogenousData = (b1.mExogenousDataFrame , b1.mExogenousVariables) 
    lEngine.train(df1 , b1.mTimeVar , lSignal, H, lExogenousData);
    lEngine.getModelInfo();
    lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution
    lEngine.standardPlots(name = "outputs/my_")

    dfapp_in = df1.copy();
    dfapp_in.tail()

    dfapp_out = lEngine.forecast(dfapp_in, H);
    dfapp_out.tail(2 * H)
    print("Forecast Columns " , dfapp_out.columns);
    Forecast_DF = dfapp_out[[b1.mTimeVar , lSignal, lSignal + '_Forecast']]
    print(Forecast_DF.info())
    print("Forecasts\n" , Forecast_DF.tail(H).values);
    
    print("\n\n<ModelInfo>")
    print(lEngine.to_json());
    print("</ModelInfo>\n\n")
    print("\n\n<Forecast>")
    print(Forecast_DF.tail(2*H).to_json(date_format='iso'))
    print("</Forecast>\n\n")

