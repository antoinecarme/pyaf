import pyaf.Bench.TS_datasets as tsds
import tests.artificial.process_artificial_dataset as art


import pandas as pd
import numpy as np
import pyaf.ForecastEngine as autof

import logging
import logging.config


np.random.seed(1789)

#logging.config.fileConfig('logging.conf')

logging.basicConfig(level=logging.INFO)

# SCIKIT_MODEL_FIT_FAILURE _Signal_0.5_LinearTrend_residue_Seasonal_DayOfWeek_residue_ARX(253) (800, 200) SVD did not converge



def process_dataset_with_noise(idataset , sigma):
    
    import warnings

    with warnings.catch_warnings():
        # warnings.simplefilter("error")
        N = idataset.mFullDataset.shape[0];
        idataset.mFullDataset[idataset.mSignalVar] = idataset.mFullDataset[idataset.mName] 
        idataset.mFullDataset["orig_" + idataset.mSignalVar] = idataset.mFullDataset[idataset.mSignalVar];
        lSignalVar = idataset.mSignalVar + "_" + str(sigma);
        lNoise = np.random.randn(N) * sigma;
        H = idataset.mHorizon[idataset.mSignalVar];
        idataset.mFullDataset[lSignalVar] = idataset.mFullDataset["orig_" + idataset.mSignalVar] + lNoise;
        idataset.mPastData = idataset.mFullDataset[:-H];
        idataset.mFutureData = idataset.mFullDataset.tail(H);
        training_ds = idataset.mPastData
        # #df.to_csv("outputs/rand_exogenous.csv")
    
    
        # N = df.shape[0];
        # df1 = df;
        lEngine = autof.cForecastEngine()
        # lEngine.mOptions.mEnableSeasonals = False;
        # lEngine.mOptions.mDebugCycles = True;
        # lEngine.mOptions.enable_slow_mode();
        # mDebugProfile = True;
        # lEngine
        lEngine.mOptions.set_active_transformations(['None']);
        lEngine.mOptions.set_active_trends(['LinearTrend']);
        lEngine.mOptions.set_active_periodics(['Seasonal_DayOfWeek']);
        lEngine.mOptions.set_active_autoregressions(['ARX']);
        lExogenousData = (idataset.mExogenousDataFrame , idataset.mExogenousVariables) 
        lEngine.train(training_ds , idataset.mTimeVar , lSignalVar, H, lExogenousData);
        lEngine.getModelInfo();
        # lEngine.standardPlots(name = "outputs/my_exog_" + str(nbex) + "_" + str(n));
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

        # lEngine.standardPlots(name = "outputs/artificial_" + idataset.mName + "_" + str(sigma))

dataset = tsds.generate_random_TS(N = 1024 , FREQ = 'D', seed = 0, trendtype = "constant", cycle_length = 12, transform = "log", sigma = 0.0, exog_count = 100, ar_order = 12);

process_dataset_with_noise(dataset, 0.5);
