
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

def get_transform_real_name(tranf):
    lDict = {
        "" : "None",
        "diff" : "Difference" ,
        "cumsum" : "Integration"
    }
    return lDict.get(tranf , "None")

def get_trend_real_name(trend):
    lDict = {
        "constant" : 'ConstantTrend',
        "linear" : 'LinearTrend',
        "poly" : 'PolyTrend'
    }
    return lDict.get(trend, 'ConstantTrend')
    
def process_dataset(N, FREQ, seed, trendtype ,
                    cycle_length, transform,
                    sigma, exog_count, ar_order):
    import pyaf.Bench.TS_datasets as tsds
    dataset = tsds.generate_random_TS(N, FREQ, seed,
                                      trendtype, cycle_length,
                                      transform, sigma, exog_count, ar_order);
    dataset.mFullDataset['Signal'] = dataset.mFullDataset[dataset.mName]
    transform1 = get_transform_real_name(transform)
    trend = get_trend_real_name(trendtype)
    model_type = (transform1, trend, "BestCycle" , ["AR", "ARX"])
    return process_dataset_1(dataset , model_type, debug = True)


def process_dataset_1(idataset, model_type, debug=False):
    idataset.mFullDataset["orig_" + idataset.mSignalVar] = idataset.mFullDataset[idataset.mSignalVar];

    return process_dataset_with_noise(idataset, model_type, 0.01 , debug);

def process_dataset_with_noise(idataset , model_type, sigma, debug=False):
    np.random.seed(1789)
    
    import warnings

    with warnings.catch_warnings():
        # warnings.simplefilter("error")
        N = idataset.mFullDataset.shape[0];
        lSignalVar = idataset.mSignalVar + "_" + str(sigma);
        lHorizon = idataset.mHorizon[idataset.mSignalVar]
        lNoise = np.random.randn(N) * sigma;
        idataset.mFullDataset[lSignalVar] = idataset.mFullDataset["orig_" + idataset.mSignalVar] + lNoise;
        idataset.mPastData = idataset.mFullDataset[:-lHorizon];
        idataset.mFutureData = idataset.mFullDataset.tail(lHorizon);
        training_ds = idataset.mPastData
        # training_ds.to_csv("/tmp/to_del_train.csv")
    
        H = lHorizon;
    
        # N = df.shape[0];
        # df1 = df;
        lEngine = autof.cForecastEngine()
        lEngine.mOptions.mDebugProfile = True;
        lEngine.mOptions.mDebugAR = True;
        lEngine.mOptions.mDebug = debug;
        lEngine.mOptions.set_active_transformations([model_type[0]])
        lEngine.mOptions.set_active_trends([model_type[1]])
        lEngine.mOptions.set_active_periodics([model_type[2]])
        lEngine.mOptions.set_active_autoregressions(model_type[3])
        # lEngine.mOptions.enable_slow_mode();
        # mDebugProfile = True;
        # lEngine
        lExogenousData = None
        if(idataset.mExogenousDataFrame is not None):
            lExogenousData = (idataset.mExogenousDataFrame , idataset.mExogenousVariables)
            
        lEngine.train(training_ds , idataset.mTimeVar , lSignalVar, H, lExogenousData);
        lEngine.getModelInfo();
        # lEngine.standardPlots(name = "outputs/my_artificial_" + idataset.mName + "_" + str(sigma));
        # lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution

        lEngine2 = pickleModel(lEngine)
        
        dfapp_in = training_ds.copy();
        dfapp_in.tail()

    
        dfapp_out = lEngine2.forecast(dfapp_in, H);
        # dfapp_out.to_csv("/tmp/to_del.csv")
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

        # lEngine2.standardPlots(name = "outputs/artificial_" + idataset.mName + "_" + str(sigma))

        return (lEngine, dfapp_in, dfapp_out)
