import numpy as np
import pandas as pd
import pyaf.ForecastEngine as autof

# the goal of these tests is to make pyaf as robust as possible against very small/bad datasets
# pyaf should automatically produce reasonable/naive/trivial models in these cases.
# it should not fail in any case (normal behavior expected)

def test_fake_model_1_row(iHorizon_train , iHorizon_apply):
    # one row dataset => always constant forecast
    df = pd.DataFrame([[0 , 0.54543]], columns = ['date' , 'signal'])
    lEngine = autof.cForecastEngine()
    lEngine.train(df , 'date' , 'signal', iHorizon_train);
    # print(lEngine.mSignalDecomposition.mBestModel.mTimeInfo.info())
    print(lEngine.mSignalDecomposition.mBestModel.getFormula())
    lPerf_H = lEngine.mSignalDecomposition.mBestModel.mForecastPerfs["signal_Forecast_" + str(iHorizon_train)]
    print("PERFS_MAPE_MASE", lPerf_H.mMAPE, lPerf_H.mMASE, )
    
    # print(df.head())
    df1 = lEngine.forecast(df , iHorizon_apply)
    # print(df1.columns)
    Forecast_DF = df1[['date' , 'signal', 'signal' + '_Forecast', 'signal_Residue', 'signal_Forecast_Lower_Bound',
       'signal_Forecast_Upper_Bound']]
    # print(Forecast_DF.info())
    print("Forecasts\n" , Forecast_DF.tail(iHorizon_apply));

    
    
def test_fake_model_2_rows(iHorizon_train , iHorizon_apply):
    # one row dataset => always constant forecast
    df = pd.DataFrame([[0 , 0.54543] , [1 , 0.43]], columns = ['date' , 'signal'])
    lEngine = autof.cForecastEngine()
    lEngine.train(df , 'date' , 'signal', iHorizon_train);
    # print(lEngine.mSignalDecomposition.mBestModel.mTimeInfo.info())
    print(lEngine.mSignalDecomposition.mBestModel.getFormula())
    lPerf_H = lEngine.mSignalDecomposition.mBestModel.mForecastPerfs["signal_Forecast_" + str(iHorizon_train)]
    print("PERFS_MAPE_MASE", lPerf_H.mMAPE, lPerf_H.mMASE, )
    # print(df.head())
    df1 = lEngine.forecast(df , iHorizon_apply)
    # print(df1.columns)
    Forecast_DF = df1[['date' , 'signal', 'signal' + '_Forecast', 'signal_Residue', 'signal_Forecast_Lower_Bound',
       'signal_Forecast_Upper_Bound']]
    # print(Forecast_DF.info())
    print("Forecasts\n" , Forecast_DF.tail(iHorizon_apply));




# In[3]:

test_fake_model_1_row( 2, 1)


# In[4]:

test_fake_model_1_row( 1, 2)


# In[5]:

test_fake_model_1_row( 2, 10)


# In[6]:

test_fake_model_1_row( 20, 10)


# In[7]:

test_fake_model_2_rows( 1, 4)


# In[8]:

test_fake_model_2_rows( 6, 2)


# In[9]:

test_fake_model_2_rows( 6, 1)


# In[10]:

test_fake_model_2_rows( 1 , 7)
