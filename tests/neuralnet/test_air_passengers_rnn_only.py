import pandas as pd
import numpy as np

import ForecastEngine as autof
import Bench.TS_datasets as tsds

import CodeGen.TS_CodeGenerator as tscodegen

import logging
import logging.config

#logging.config.fileConfig('logging.conf')

logging.basicConfig(level=logging.INFO)



b1 = tsds.load_airline_passengers()
df = b1.mPastData

df.head()


lEngine = autof.cForecastEngine()
lEngine

H = b1.mHorizon;
# lEngine.mOptions.enable_slow_mode();
lEngine.mOptions.mDebugPerformance = True;
lEngine.mOptions.mParallelMode = True;
lEngine.mOptions.mEnableARModels = True;
lEngine.mOptions.mEnableARXModels = False;
lEngine.mOptions.mEnableRNNModels = True;
lEngine.train(df , b1.mTimeVar , b1.mSignalVar, H);
lEngine.getModelInfo();
print(lEngine.mSignalDecomposition.mTrPerfDetails.head());

lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution

lEngine.standrdPlots(name = "outputs/rnn_my_airline_passengers")

dfapp_in = df.copy();
dfapp_in.tail()

#H = 12
dfapp_out = lEngine.forecast(dfapp_in, H);
dfapp_out.tail(2 * H)
print("Forecast Columns " , dfapp_out.columns);
lForecastColumnName = b1.mSignalVar + '_Forecast'
Forecast_DF = dfapp_out[[b1.mTimeVar , b1.mSignalVar, lForecastColumnName , lForecastColumnName + '_Lower_Bound',  lForecastColumnName + '_Upper_Bound' ]]
print(Forecast_DF.info())
print("Forecasts\n" , Forecast_DF.tail(2*H));

print("\n\n<ModelInfo>")
print(lEngine.to_json());
print("</ModelInfo>\n\n")
print("\n\n<Forecast>")
print(Forecast_DF.to_json(date_format='iso'))
print("</Forecast>\n\n")

# lEngine.standrdPlots(name = "outputs/rnn_airline_passengers")
