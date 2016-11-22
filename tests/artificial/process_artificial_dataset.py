
import pandas as pd
import numpy as np
import pyaf.ForecastEngine as autof

import pyaf.CodeGen.TS_CodeGenerator as tscodegen

def process_dataset(idataset):
    idataset.mFullDataset["orig_" + idataset.mSignalVar] = idataset.mFullDataset[idataset.mSignalVar];

    for i in range(10):
        sig = 0.1 * i;
        process_dataset_with_noise(idataset, sig);

def process_dataset_with_noise(idataset , sigma):
    
    import warnings

    with warnings.catch_warnings():
        # warnings.simplefilter("error")
        N = idataset.mFullDataset.shape[0];
        lSignalVar = idataset.mSignalVar + "_" + str(sigma);
        lNoise = np.random.randn(N) * sigma;
        idataset.mFullDataset[lSignalVar] = idataset.mFullDataset["orig_" + idataset.mSignalVar] + lNoise;
        idataset.mPastData = idataset.mFullDataset[:-idataset.mHorizon];
        idataset.mFutureData = idataset.mFullDataset.tail(idataset.mHorizon);
        training_ds = idataset.mPastData
        # df.to_csv("outputs/rand_exogenous.csv")
    
        H = idataset.mHorizon;
    
        # N = df.shape[0];
        # df1 = df;
        lEngine = autof.cForecastEngine()
        # lEngine.mOptions.mEnableSeasonals = False;
        # lEngine.mOptions.mDebugCycles = True;
        # lEngine.mOptions.enable_slow_mode();
        # mDebugProfile = True;
        # lEngine
        lExogenousData = (idataset.mExogenousDataFrame , idataset.mExogenousVariables) 
        lEngine.train(training_ds , idataset.mTimeVar , lSignalVar, H, lExogenousData);
        lEngine.getModelInfo();
        # lEngine.standrdPlots(name = "outputs/my_exog_" + str(nbex) + "_" + str(n));
        # lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution
        
        dfapp_in = training_ds.copy();
        dfapp_in.tail()

    
        dfapp_out = lEngine.forecast(dfapp_in, H);
        dfapp_out.tail(2 * H)
        print("Forecast Columns " , dfapp_out.columns);
        Forecast_DF = dfapp_out[[idataset.mTimeVar , lSignalVar, lSignalVar + '_Forecast']]
        print(Forecast_DF.info())
        print("Forecasts\n" , Forecast_DF.tail(H).values);
        
        print("\n\n<ModelInfo>")
        print(lEngine.to_json());
        print("</ModelInfo>\n\n")
        print("\n\n<Forecast>")
        print(Forecast_DF.tail(2*H).to_json(date_format='iso'))
        print("</Forecast>\n\n")
