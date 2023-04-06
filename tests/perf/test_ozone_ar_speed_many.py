import pandas as pd
import numpy as np
import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds

#get_ipython().magic('matplotlib inline')

import datetime as dt
from dateutil.relativedelta import relativedelta

b1 = tsds.load_ozone()

def addWeeks(iTime , iWeeks):
    lTime = iTime
    date_after_week = lTime + relativedelta(weeks=iWeeks)
    return np.datetime64(date_after_week) 

def replicate(df , K):
    N = df.shape[0];
    y = pd.Series(range(K*N));
    tmin = df.Time.min()
    Time1 = y.apply(lambda x : addWeeks(tmin , x))
    # Time1.describe()
    Ozone1 = pd.concat([df.Ozone] * K)
    print(Time1.shape, Ozone1.shape, Time1.describe(), Ozone1.describe())
    df1 = pd.DataFrame();
    df1['Time'] = Time1.values;
    print(Ozone1.describe());
    df1['Ozone'] = Ozone1.values;
    return df1;


def buildModel(df, ar_order, H):

    lEngine = autof.cForecastEngine()

    lEngine.mOptions.mMaxAROrder = ar_order;
    lEngine.train(df , b1.mTimeVar , b1.mSignalVar, H);
    lEngine.getModelInfo();

    lEngine.standardPlots("outputs/perf_test_ozone_ar_speed_many_" + str(ar_order));

def run_test___disabled___():
    df = b1.mPastData    
    df1 = replicate(df, 50);
    df1.head()
    H = b1.mHorizon;
    for order in np.arange(0, 1000, 50):
        buildModel(df1 , int(order) , H)

def run_test(order):
    df = b1.mPastData    
    df1 = replicate(df, 50);
    df1.head()
    H = b1.mHorizon;
    buildModel(df1 , int(order) , H)

# disabled        
# run_test()
