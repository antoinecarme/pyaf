import pandas as pd
import numpy as np

import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds

import pyaf.CodeGen.TS_CodeGenerator as tscodegen

b1 = tsds.load_airline_passengers()
df = b1.mPastData

df.head()



H = 3;


N = 24;
for n in [N]:
    df1 = df.head(n).copy();
    lEngine = autof.cForecastEngine()
    lEngine
    lEngine.mOptions.mEnableARModels = False;
    # lEngine.mOptions.mDebugCycles = False;

    lEngine.train(df1 , b1.mTimeVar , b1.mSignalVar, H);
    lEngine.getModelInfo();
    lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution
    dfapp_in = df1.copy();
    dfapp_in.tail()

    # H = 12
    dfapp_out = lEngine.forecast(dfapp_in, H);
    dfapp_out.tail(2 * H)
    print("Forecast Columns " , dfapp_out.columns);
    lForecastColumnName = b1.mSignalVar + '_Forecast'
    Forecast_DF = dfapp_out;
    # [[b1.mTimeVar , b1.mSignalVar, lForecastColumnName , lForecastColumnName + '_Lower_Bound',  lForecastColumnName + '_Upper_Bound' ]]
    print(Forecast_DF.info())
    # print("Forecasts_HEAD\n" , Forecast_DF.head(2*H).values);
    # print("Forecasts_TAIL\n" , Forecast_DF.tail(2*H).values);
    lCodeGenerator = tscodegen.cTimeSeriesCodeGenerator();
    lSQL = lCodeGenerator.testGeneration(lEngine);
