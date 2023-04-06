import pandas as pd
import numpy as np

import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds

import logging
import logging.config

#logging.config.fileConfig('logging.conf')

logging.basicConfig(level=logging.DEBUG)

#get_ipython().magic('matplotlib inline')


def buildModel(arg):
    (cyc , freq, nbrows) = arg
        
    print("TEST_CYCLES_START", nbrows, freq, cyc)
    b1 = tsds.generate_random_TS(N = nbrows , FREQ = freq, seed = 0, trendtype = "constant", cycle_length = cyc, transform = "", sigma = 0.1, exog_count = 0);
    df = b1.mPastData
    df['Signal'] = df[b1.mName]
    df = df[[b1.mTimeVar , b1.mSignalVar]].copy()
    lSignal = 'Signal_Cycle_' + str(nbrows) + "_" + str(freq) + "_" + str(cyc)
    df.columns = [b1.mTimeVar, lSignal]
    print(df.head())
    print(df.tail())
    print(df.describe())
        
    lEngine = autof.cForecastEngine()
    lEngine.mOptions.disable_all_transformations();
    lEngine.mOptions.set_active_transformations(['None']);

    lEngine.mOptions.mCycleLengths = [ k for k in range(2,128) ];
    lEngine.mOptions.mDebugCycles = False;
    lEngine

    H = 12;
    lEngine.train(df , b1.mTimeVar , lSignal, H);
    lEngine.getModelInfo();

    lName = b1.mName
    lEngine.standardPlots("outputs/periodicities_test_" + lName);    
    lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution
    
    dfapp_in = df.copy();
    dfapp_in.tail()

    # H = 12
    dfapp_out = lEngine.forecast(dfapp_in, H);
    dfapp_out.tail(2 * H)
    print("Forecast Columns " , dfapp_out.columns);
    Forecast_DF = dfapp_out[[b1.mTimeVar , lSignal, lSignal + '_Forecast']]
    print(Forecast_DF.info())
    print("Forecasts\n" , Forecast_DF.tail(H).values);
        
    print("\n\n<ModelInfo>")
    print(lEngine.to_json());
    print("</ModelInfo>\n\n")
    print("\n\n<Forecast>")
    print(Forecast_DF.tail(H).to_json(date_format='iso'))
    print("</Forecast>\n\n")
        
    print("TEST_CYCLES_END", cyc)
    del lEngine
    del df
    del b1
    del dfapp_in
    del dfapp_out
    del Forecast_DF
        

'''
http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

Alias 	Description
B 	business day frequency
C 	custom business day frequency (experimental)
D 	calendar day frequency
W 	weekly frequency
M 	month end frequency
SM 	semi-month end frequency (15th and end of month)
BM 	business month end frequency
CBM 	custom business month end frequency
MS 	month start frequency
SMS 	semi-month start frequency (1st and 15th)
BMS 	business month start frequency
CBMS 	custom business month start frequency
Q 	quarter end frequency
BQ 	business quarter endfrequency
QS 	quarter start frequency
BQS 	business quarter start frequency
A 	year end frequency
BA 	business year end frequency
AS 	year start frequency
BAS 	business year start frequency
BH 	business hour frequency
H 	hourly frequency
T, min 	minutely frequency
S 	secondly frequency
L, ms 	milliseconds
U, us 	microseconds
N 	nanoseconds
'''

def run_in_parallel():
    lCycles = [ 5 , 7 , 12 , 15, 24, 30 , 60, 120, 360];
    lFreqs = ['S' , 'T', 'H' , 'BH', 'D' , 'B', 'W' , 'SM', 'M']

    args = [];
    for cyc in lCycles:
        for freq in lFreqs:
            lRows = [25, 50, 100, 200, 400, 1600, 3200];
            if(freq == 'W'):
                lRows = [25, 50, 100, 200, 400, 1600];
            if(freq == 'SM'):
                lRows = [25, 50, 100, 200, 400];
            if(freq == 'M'):
                lRows = [25, 50, 100, 200, 400];
            for nbrows in lRows:
                args = args + [(cyc , freq, nbrows)]

    import multiprocessing as mp
    pool = mp.Pool()
    asyncResult = pool.map_async(buildModel, args);

    resultList = asyncResult.get()


# run_in_parallel();
