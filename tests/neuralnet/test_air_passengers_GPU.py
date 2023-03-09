
def pickleModel(iModel):
    import pickle
    output = pickle.dumps(iModel)
    lReloadedObject = pickle.loads(output)
    output2 = pickle.dumps(lReloadedObject)    
    assert(iModel.to_json() == lReloadedObject.to_json())
    return lReloadedObject;

def buildModel(iParallel = True):

    import pandas as pd
    import numpy as np

    import pyaf.ForecastEngine as autof
    import pyaf.Bench.TS_datasets as tsds

    import logging
    import logging.config

    # logging.config.fileConfig('logging.conf')

    logging.basicConfig(level=logging.INFO)

    b1 = tsds.load_airline_passengers()
    df = b1.mPastData

    df.head()


    lEngine = autof.cForecastEngine()
    lEngine

    H = b1.mHorizon;
    # lEngine.mOptions.enable_slow_mode();
    lEngine.mOptions.mEnableSeasonals = True;
    lEngine.mOptions.mEnableCycles = True;
    lEngine.mOptions.mDebugPerformance = True;
    lEngine.mOptions.mParallelMode = iParallel;
    lEngine.mOptions.set_active_autoregressions(['MLP' , 'LSTM']);
    # lEngine.mOptions.mMaxAROrder = 2;
    lEngine.train(df , b1.mTimeVar , b1.mSignalVar, H);

    lEngine2 = pickleModel(lEngine)

    lEngine2.getModelInfo();
    

    lEngine2.mSignalDecomposition.mBestModel.mTimeInfo.mResolution
    
    lEngine2.standardPlots(name = "outputs/rnn_my_airline_passengers")
    
    dfapp_in = df.copy();
    dfapp_in.tail()

    # H = 12
    dfapp_out = lEngine2.forecast(dfapp_in, H);
    dfapp_out.tail(2 * H)
    print("Forecast Columns " , dfapp_out.columns);
    lForecastColumnName = b1.mSignalVar + '_Forecast'
    Forecast_DF = dfapp_out[[b1.mTimeVar , b1.mSignalVar, lForecastColumnName , lForecastColumnName + '_Lower_Bound',  lForecastColumnName + '_Upper_Bound' ]]
    print(Forecast_DF.info())
    print("Forecasts\n" , Forecast_DF.tail(2*H));
    
    print("\n\n<ModelInfo>")
    print(lEngine2.to_json());
    print("</ModelInfo>\n\n")
    print("\n\n<Forecast>")
    print(Forecast_DF.tail(2*H).to_json(date_format='iso'))
    print("</Forecast>\n\n")

    # lEngine2.standardPlots(name = "outputs/rnn_airline_passengers")
