import pandas as pd
import numpy as np
import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds


def create_df():
    df = pd.DataFrame()
    df['time'] = range(1200)
    df['rate'] = 0.2; # np.sin(df['time'])
    df.info();
    N = df.shape[0];
    df['signal2'] = 1;
    for i in range(N-1):
        df.loc[i+1, 'signal2'] = df.loc[i, 'signal2'] * (df.loc[i, 'rate'] + 1);
    print(df.head())
    return df;



def test_transformation(itransformation):
    df = create_df();
    # df.to_csv('a.csv')
    lEngine = autof.cForecastEngine()
    lEngine

    H = 12;
    # lEngine.mOptions.enable_slow_mode();
    lEngine.mOptions.mDebugPerformance = True;

    if(itransformation is not None):
        lEngine.mOptions.disable_all_transformations();
        lEngine.mOptions.set_active_transformations([itransformation]);
    lEngine.mOptions.mBoxCoxOrders = lEngine.mOptions.mExtensiveBoxCoxOrders;

    lSignalVar = 'signal2'
    lTimeVar = 'time'
    lEngine.train(df , lTimeVar , lSignalVar, H);
    lEngine.getModelInfo();

    lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution
    
    lEngine.standardPlots("outputs/my_airline_" + str(itransformation));
    
    dfapp_in = df.copy();
    dfapp_in.tail()
    
    dfapp_out = lEngine.forecast(dfapp_in, H);
    #dfapp_out.to_csv("outputs/ozone_apply_out_" + itransformation + ".csv")
    dfapp_out.tail(H)
    print("Forecast Columns " , dfapp_out.columns);
    Forecast_DF = dfapp_out[[lTimeVar , lSignalVar, lSignalVar + '_Forecast']]
    print(Forecast_DF.info())
    print("Forecasts\n" , Forecast_DF.tail(H));

    print("\n\n<ModelInfo>")
    print(lEngine.to_json());
    print("</ModelInfo>\n\n")
    print("\n\n<Forecast>")
    print(Forecast_DF.tail(2*H).to_json(date_format='iso'))
    print("</Forecast>\n\n")


test_transformation('RelativeDifference')
test_transformation(None)
