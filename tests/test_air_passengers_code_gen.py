import pandas as pd
import numpy as np
import Bench.TS_datasets as tsds
import AutoForecast as autof
import TS_CodeGenerator as tscodegen

b1 = tsds.load_airline_passengers()
df = b1.mPastData

df.head()



H = b1.mHorizon;


N = df.shape[0];
for n in range(2*H,  N , 10):
    df1 = df.head(n).copy();
    lAutoF = autof.cAutoForecast()
    lAutoF
    lAutoF.train(df1 , b1.mTimeVar , b1.mSignalVar, H);
    lAutoF.getModelInfo();
    lAutoF.mSignalDecomposition.mBestTransformation.mTimeInfo.mResolution
    dfapp_in = df1.copy();
    dfapp_in.tail()

    # H = 12
    dfapp_out = lAutoF.forecast(dfapp_in, H);
    dfapp_out.tail(2 * H)
    print("Forecast Columns " , dfapp_out.columns);
    lForecastColumnName = b1.mSignalVar + '_BestModelForecast'
    Forecast_DF = dfapp_out;
    # [[b1.mTimeVar , b1.mSignalVar, lForecastColumnName , lForecastColumnName + '_Lower_Bound',  lForecastColumnName + '_Upper_Bound' ]]
    print(Forecast_DF.info())
    print("Forecasts_HEAD\n" , Forecast_DF.head(2*H).values);
    print("Forecasts_TAIL\n" , Forecast_DF.tail(2*H).values);
    lCodeGenerator = tscodegen.cTimeSeriesCodeGenerator();
    lSQL = lCodeGenerator.testGeneration(lAutoF);
