# from __future__ import absolute_import

import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds
import warnings



def process_dataset(idataset):
    
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        N = idataset.mFullDataset.shape[0];
        lSignalVar = idataset.mSignalVar;
        idataset.mFullDataset[idataset.mSignalVar] = idataset.mFullDataset[idataset.mName] 

        H = 2;

        idataset.mPastData = idataset.mFullDataset[:-H];
        idataset.mFutureData = idataset.mFullDataset.tail(H);
        training_ds = idataset.mPastData
        # #df.to_csv("outputs/rand_exogenous.csv")
    
    
        # N = df.shape[0];
        # df1 = df;
        lEngine = autof.cForecastEngine()
        # lEngine.mOptions.mEnableSeasonals = False;
        # lEngine.mOptions.mDebugCycles = False;
        # lEngine.mOptions.enable_slow_mode();
        # mDebugProfile = True;
        # lEngine
        lExogenousData = None
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


with warnings.catch_warnings():
    warnings.simplefilter("error")

    dataset = tsds.generate_random_TS(N = 40 , FREQ = 'D', seed = 0, trendtype = "linear", cycle_length = 4, transform = "exp", sigma = 2.0, exog_count = 0);

    process_dataset(dataset);
