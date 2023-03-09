import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=1.0,exception_verbosity=high,dnn.enabled=False,optimizer=None"
os.environ["KERAS_BACKEND"] = "tensorflow"

import pandas as pd
import numpy as np

import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds

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
lEngine.mOptions.mEnableSeasonals = False;
lEngine.mOptions.mEnableCycles = False;
lEngine.mOptions.mDebugPerformance = True;
lEngine.mOptions.mParallelMode = False;

lEngine.mOptions.disable_all_transformations();
lEngine.mOptions.disable_all_trends();
lEngine.mOptions.disable_all_periodics();
lEngine.mOptions.disable_all_autoregressions();
lEngine.mOptions.set_active_autoregressions(['LSTM']);
lEngine.mOptions.mMaxAROrder = 7;

lEngine.train(df , b1.mTimeVar , b1.mSignalVar, H);
lEngine.getModelInfo();


lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution

lEngine.standardPlots(name = "outputs/rnn_tf_my_airline_passengers")

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
print(Forecast_DF.tail(2*H).to_json(date_format='iso'))
print("</Forecast>\n\n")

# lEngine.standardPlots(name = "outputs/rnn_airline_passengers")
