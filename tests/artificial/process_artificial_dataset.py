
import pandas as pd
import numpy as np
import pyaf.ForecastEngine as autof

import logging
import logging.config

#logging.config.fileConfig('logging.conf')

logging.basicConfig(level=logging.INFO)


def pickleModel(iModel):
    import pickle
    output = pickle.dumps(iModel)
    lReloadedObject = pickle.loads(output)
    output2 = pickle.dumps(lReloadedObject)    
    assert(iModel.to_json() == lReloadedObject.to_json())
    return lReloadedObject;

def process_dataset(idataset, debug=False):
    idataset.mFullDataset["orig_" + idataset.mSignalVar] = idataset.mFullDataset[idataset.mSignalVar];

    process_dataset_with_noise(idataset, 0.1 , debug);

def process_dataset_with_noise(idataset , sigma, debug=False):
    
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        N = idataset.mFullDataset.shape[0];
        lSignalVar = idataset.mSignalVar + "_" + str(sigma);
        lHorizon = idataset.mHorizon[idataset.mSignalVar]
        lNoise = np.random.randn(N) * sigma;
        idataset.mFullDataset[lSignalVar] = idataset.mFullDataset["orig_" + idataset.mSignalVar] + lNoise;
        idataset.mPastData = idataset.mFullDataset[:-lHorizon];
        idataset.mFutureData = idataset.mFullDataset.tail(lHorizon);
        training_ds = idataset.mPastData
        # #df.to_csv("outputs/rand_exogenous.csv")
    
        H = lHorizon;
    
        # N = df.shape[0];
        # df1 = df;
        lEngine = autof.cForecastEngine()
        # lEngine.mOptions.mEnableSeasonals = False;
        lEngine.mOptions.mDebug = debug;
        # lEngine.mOptions.enable_slow_mode();
        # mDebugProfile = True;
        # lEngine
        lExogenousData = (idataset.mExogenousDataFrame , idataset.mExogenousVariables) 
        lEngine.train(training_ds , idataset.mTimeVar , lSignalVar, H, lExogenousData);
        lEngine.getModelInfo();
        # lEngine.standrdPlots(name = "outputs/my_artificial_" + idataset.mName + "_" + str(sigma));
        # lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution

        lEngine2 = pickleModel(lEngine)
        
        dfapp_in = training_ds.copy();
        dfapp_in.tail()

    
        dfapp_out = lEngine2.forecast(dfapp_in, H);
        dfapp_out.tail(2 * H)
        print("Forecast Columns " , dfapp_out.columns);
        lForecastName = lSignalVar + '_Forecast'
        Forecast_DF = dfapp_out[[idataset.mTimeVar , lSignalVar,
                                 lForecastName ,
                                 lForecastName + "_Lower_Bound",
                                 lForecastName + "_Upper_Bound"]]
        Forecast_DF.info()
        print("Forecasts\n" , Forecast_DF.tail(H).values);
        
        print("\n\n<ModelInfo>")
        print(lEngine2.to_json());
        print("</ModelInfo>\n\n")
        print("\n\n<Forecast>")
        print(Forecast_DF.tail(2*H).to_json(date_format='iso'))
        print("</Forecast>\n\n")

        # lEngine2.standrdPlots(name = "outputs/artificial_" + idataset.mName + "_" + str(sigma))
