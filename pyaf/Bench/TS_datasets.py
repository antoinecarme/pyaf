# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license


import pandas as pd
import numpy as np
import datetime
from datetime import date

# from memory_profiler import profile

import os.path
from . import stocks_symbol_list as symlist


class cTimeSeriesDatasetSpec:

    def __init__(self):
        self.mSignalFrame = pd.DataFrame()
        self.mExogenousDataFrame = None;
        self.mExogenousVariables = None;
        self.mHierarchy = None;
        
    def getName(self):
        return self.mName;

    def getTimeVar(self):
        return self.mTimeVar;

    def getSignalVar(self):
        return self.mSignalVar;

    def getDescription(self):
        return self.mDescription;

    def getPastData(self):
        return self.mPastData;

    def getFutureData(self):
        return self.mFutureData;

    def getHorizon(self):
        return self.mHorizon;

    
# @profile    
def load_airline_passengers() :
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "AirLine_Passengers"
    tsspec.mDescription = "AirLine Passengers"

    trainfile = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/AirPassengers.csv"
    cols = ["ID" , "time", "AirPassengers"];
    df_train = pd.read_csv(trainfile, names = cols, sep=r',', index_col='ID', engine='python', skiprows=1);

    tsspec.mTimeVar = "time";
    tsspec.mSignalVar = "AirPassengers";
    tsspec.mHorizon = 12;
    tsspec.mPastData = df_train[:-tsspec.mHorizon];
    tsspec.mFutureData = df_train.tail(tsspec.mHorizon);
    
    return tsspec


    
# @profile    
def load_cashflows() :
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "CashFlows"
    tsspec.mDescription = "CashFlows dataset"

    trainfile = "https://raw.githubusercontent.com/antoinecarme/pyaf/master/data/CashFlows.txt"
    tsspec.mFullDataset = pd.read_csv(trainfile, sep=r'\t', engine='python');
    tsspec.mFullDataset['Date'] = tsspec.mFullDataset['Date'].apply(lambda x : datetime.datetime.strptime(x, "%Y-%m-%d"))
    tsspec.mFullDataset = tsspec.mFullDataset.head(251)

    tsspec.mTimeVar = "Date";
    tsspec.mSignalVar = "Cash";
    tsspec.mHorizon = 21;
    tsspec.mPastData = tsspec.mFullDataset[:-tsspec.mHorizon];
    tsspec.mFutureData = tsspec.mFullDataset.tail(tsspec.mHorizon);
    
    return tsspec

#ozone-la.txt
#https://datamarket.com/data/set/22u8/ozon-concentration-downtown-l-a-1955-1972#


def to_date(idatetime):
    d = datetime.date(idatetime.year, idatetime.month, idatetime.day);
    return d;

# @profile    
def load_ozone() :
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "Ozone"
    tsspec.mDescription = "https://datamarket.com/data/set/22u8/ozon-concentration-downtown-l-a-1955-1972"
    
    #trainfile = "data/ozone-la.csv"
    trainfile = "https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/ozone-la.csv"

    cols = ["Month" , "Ozone"];
    
    df_train = pd.read_csv(trainfile, names = cols, sep=r',', engine='python', skiprows=1);
    df_train['Time'] = df_train['Month'].apply(lambda x : datetime.datetime.strptime(x, "%Y-%m"))
    lType = 'datetime64[D]';
    # df_train['Time'] = df_train['Time'].apply(to_date).astype(lType);
    print(df_train.head());
    
    tsspec.mTimeVar = "Time";
    tsspec.mSignalVar = "Ozone";
    tsspec.mHorizon = 12;
    tsspec.mPastData = df_train[:-tsspec.mHorizon];
    tsspec.mFutureData = df_train.tail(tsspec.mHorizon);
    
    return tsspec

# @profile    
def load_ozone_exogenous() :
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "Ozone"
    tsspec.mDescription = "https://datamarket.com/data/set/22u8/ozon-concentration-downtown-l-a-1955-1972"
    
    #trainfile = "data/ozone-la.csv"
    trainfile = "https://raw.githubusercontent.com/antoinecarme/pyaf/master/data/ozone-la-exogenous.csv"
    # "https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/ozone-la.csv"

    cols = ["Date", "Month", "Exog2", "Exog3", "Exog4", "Ozone"];
    
    df_train = pd.read_csv(trainfile, names = cols, sep=r',', engine='python', skiprows=1);
    df_train['Time'] = df_train['Date'].apply(lambda x : datetime.datetime.strptime(x, "%Y-%m"))

    tsspec.mTimeVar = "Time";
    tsspec.mSignalVar = "Ozone";
    tsspec.mExogenousVariables = ["Month", "Exog2", "Exog3", "Exog4"];
    # this is the full dataset . must contain future exogenius data
    tsspec.mExogenousDataFrame = df_train;
    # tsspec.mExogenousVariables = ["Exog2"];
    tsspec.mHorizon = 12;
    tsspec.mPastData = df_train[:-tsspec.mHorizon];
    tsspec.mFutureData = df_train.tail(tsspec.mHorizon);


    print(df_train.head())
    return tsspec


def load_ozone_exogenous_categorical() :
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "Ozone"
    tsspec.mDescription = "https://datamarket.com/data/set/22u8/ozon-concentration-downtown-l-a-1955-1972"
    
    #trainfile = "data/ozone-la.csv"
    trainfile = "https://raw.githubusercontent.com/antoinecarme/pyaf/master/data/ozone-la-exogenous.csv"
    # "https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/ozone-la.csv"

    cols = ["Date", "Month", "Exog2", "Exog3", "Exog4", "Ozone"];
    
    df_train = pd.read_csv(trainfile, names = cols, sep=r',', engine='python', skiprows=1);
    df_train['Time'] = df_train['Date'].apply(lambda x : datetime.datetime.strptime(x, "%Y-%m"))

    for col in ["Exog2", "Exog3", "Exog4"]:
        categs = sorted(df_train[col].unique())
        cat_type = pd.api.types.CategoricalDtype(categories=categs, ordered=True)
        df_train[col] = df_train[col].astype(cat_type)

    ozone_shifted_2 = df_train.shift(2)
    ozone_shifted_1 = df_train.shift(1)
    lSig1 =  df_train['Ozone'] * (ozone_shifted_2['Exog3'] == 'AW') +  df_train['Ozone'] * (ozone_shifted_1['Exog3'] == 'AX') 
    lSig2 =  df_train['Ozone'] * (ozone_shifted_1['Exog2'] >= 4)
    lSig3 =  df_train['Ozone'] * (ozone_shifted_1['Exog4'] <= 'P_S')
    df_train['Ozone2'] = lSig1 + lSig2 + lSig3
    tsspec.mTimeVar = "Time";
    tsspec.mSignalVar = "Ozone2";
    tsspec.mExogenousVariables = ["Exog2", "Exog3", "Exog4"];
    # this is the full dataset . must contain future exogenius data
    tsspec.mExogenousDataFrame = df_train;
    # tsspec.mExogenousVariables = ["Exog2"];
    tsspec.mHorizon = 12;
    tsspec.mPastData = df_train[:-tsspec.mHorizon];
    tsspec.mFutureData = df_train.tail(tsspec.mHorizon);


    print(df_train.head())
    return tsspec




def add_some_noise(x , p , min_sig, max_sig, e , f):
    delta = (x - min_sig) / (max_sig - min_sig);
    if( (delta >= e) and (delta <= f) ):
        if(np.random.random() < p):
            return "A";
    return "0";


def gen_trend(N , trendtype):
    lTrend = pd.Series(dtype='float64');
    a = (2 * np.random.random() - 1);
    b = (2 * np.random.random() - 1);
    c = (2 * np.random.random() - 1);
    print("TREND" , a , b ,c);
    lTrend = 0
    if(trendtype == "ConstantTrend"):
        lTrend = a
    elif(trendtype == "LinearTrend"):
        x = np.arange(0,N) / N ;
        lTrend =  a * x + b;
    elif(trendtype == "PolyTrend"):
        x = np.arange(0,N) / N;
        lTrend =  a * x * x + b * x + c;
    # lTrend.plot();
    return lTrend;

def gen_cycle(N , cycle_length):
    lCycle = pd.Series(dtype='float64');
    if(cycle_length > 0):
        lCycle = np.arange(0,N) % cycle_length;
        lValues = np.random.randint(0, cycle_length, size=(cycle_length, 1)) /cycle_length;
        lCycle = pd.Series(lCycle).apply(lambda x : lValues[int(x)][0]);
    if(cycle_length == 0):
        lCycle = 0;
    return lCycle;
    
def gen_ar(N , ar_order):
    lAR = pd.Series(dtype='float64');
    if(ar_order > 0):
        lSig = pd.Series(np.arange(0, N) / N);
        lAR = 0;
        a_p = 1;
        for p in range(1 , ar_order+1):
            a_p = a_p * np.random.uniform();
            lAR = lSig.shift(p).fillna(0) * a_p + lAR;
    if(ar_order == 0):
        lAR = 0;
    return lAR;

def apply_old_transform(signal , transform):
    transformed = None
    if(transform == "exp"):
        transformed = np.exp(-signal)
    if(transform == "log"):
        transformed = np.log(signal)
    if(transform == "sqrt"):
        transformed = np.sqrt(signal)
    if(transform == "sqr"):
        transformed = np.power(signal , 2)
    if(transform == "pow3"):
        transformed = np.power(signal , 3)
    if(transform == "inv"):
        transformed = 1.0 / (signal)
    if(transform == "diff"):
        transformed = signal - signal.shift(1).fillna(0.0);
    if(transform == "cumsum"):
        transformed = signal.cumsum();
    return transformed
    
def apply_transform(signal , transform):
    import pyaf.TS.Signal_Transformation as tstransf
    arg = None
    if(transform == "Quantization"):
        arg = 10
    if(transform == "BoxCox"):
        arg = 0
    tr = tstransf.create_tranformation(transform , arg)
    transformed = None
    if(tr is None):
        transformed = apply_old_transform(signal, transform)
    else :
        tr.fit(signal)
        transformed = tr.invert(signal)
        # print(signal.head())
        # print(transformed.head())
    return transformed

def generate_random_TS(N , FREQ, seed, trendtype, cycle_length, transform, sigma = 1.0, exog_count = 20, ar_order = 0) :
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "Signal_" + str(N) + "_" + str(FREQ) +  "_" + str(seed)  + "_" + str(trendtype) +  "_" + str(cycle_length)   + "_" + str(transform)   + "_" + str(sigma) + "_" + str(exog_count) ;
    print("GENERATING_RANDOM_DATASET" , tsspec.mName);
    tsspec.mDescription = "Random generated dataset";

    np.random.seed(seed);

    df_train = pd.DataFrame();
    #df_train['Date'] = np.arange(0,N)

    '''
    http://pandas.pydata.org/pandas-docs/stable/timeseries.html
    DateOffset objects

    In the preceding examples, we created DatetimeIndex objects at various frequencies by passing in frequency strings
    like "M", "W", and "BM" to the freq keyword. Under the hood, these frequency strings are being translated into an
    instance of pandas DateOffset, which represents a regular frequency increment.
    Specific offset logic like "month", "business day", or "one hour" is represented in its various subclasses.
    '''
    df_train['Date'] = pd.date_range('2000-1-1', periods=N, freq=FREQ)
    df_train['GeneratedTrend'] = gen_trend(N , trendtype);

    df_train['GeneratedCycle'] = gen_cycle(N , cycle_length);

    df_train['GeneratedAR'] = gen_ar(N , ar_order);

    df_train['Noise'] = np.random.randn(N, 1) * sigma;
    df_train['Signal'] = 100 * df_train['GeneratedTrend'] +  10 * df_train['GeneratedCycle'] + 1 * df_train['GeneratedAR'] + df_train['Noise']

    min_sig = df_train['Signal'].min();
    max_sig = df_train['Signal'].max();
    # print(df_train.info())
    tsspec.mExogenousVariables = [];
    tsspec.mExogenousDataFrame = pd.DataFrame();
    tsspec.mExogenousDataFrame['Date'] = df_train['Date']
    for e in range(exog_count):
        label = "exog_" + str(e+1);
        tsspec.mExogenousDataFrame[label] = df_train['Signal'].apply(
            lambda x : add_some_noise(x , 0.1 , 
                                      min_sig, 
                                      max_sig, 
                                      e/exog_count ,
                                      (e+3)/exog_count ));
        tsspec.mExogenousVariables = tsspec.mExogenousVariables + [ label ];

    # print(tsspec.mExogenousDataFrame.info())

    # this is the full dataset . must contain future exogenius data
    pos_signal = df_train['Signal'] - min_sig + 1.0;

    df_train['Signal'] = apply_transform(pos_signal , transform)

    # df_train.to_csv(tsspec.mName + ".csv");

    tsspec.mTimeVar = "Date";
    tsspec.mSignalVar = "Signal";
    lHorizon = min(12, max(1, N // 30));
    tsspec.mHorizon = {}
    tsspec.mHorizon[tsspec.mSignalVar] = lHorizon
    tsspec.mHorizon[tsspec.mName] = lHorizon
    tsspec.mFullDataset = df_train;
    tsspec.mFullDataset[tsspec.mName] = tsspec.mFullDataset['Signal'];
    tsspec.mPastData = df_train[:-lHorizon];
    tsspec.mFutureData = df_train.tail(lHorizon);
    
    return tsspec




# @profile    
def load_NN5():
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "NN5";
    tsspec.mDescription = "NN5 competition final dataset";
    trainfile = "https://raw.githubusercontent.com/antoinecarme/pyaf/master/data/NN5-Final-Dataset.csv"
    tsspec.mFullDataset = pd.read_csv(trainfile, sep='\t', header=0, engine='python');
    tsspec.mFullDataset['Day'] = tsspec.mFullDataset['Day'].apply(lambda x : datetime.datetime.strptime(x, "%d-%b-%y"))
    # tsspec.mFullDataset = tsspec.mFullDataset
    # tsspec.mFullDataset.fillna(method = "ffill", inplace = True);
    tsspec.mHorizon = {};
    for sig in tsspec.mFullDataset.columns:
        tsspec.mHorizon[sig] = 56
#    df_test = tsspec.mFullDataset.tail(tsspec.mHorizon);
    df_train = tsspec.mFullDataset;

    tsspec.mTimeVar = "Day";
    tsspec.mSignalVar = "Signal";
    # tsspec.mPastData = df_train[:-tsspec.mHorizon];
    # tsspec.mFutureData = df_train.tail(tsspec.mHorizon);
    
    return tsspec


# @profile    
def load_NN3_part1():
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "NN3_Part1";
    tsspec.mDescription = "NN3 competition final dataset - part 1";
    trainfile = "https://raw.githubusercontent.com/antoinecarme/pyaf/master/data/NN3-Final-Dataset-part1.csv"
    tsspec.mFullDataset = pd.read_csv(trainfile, sep='\t', header=0, engine='python');
    tsspec.mFullDataset['Date'] = np.arange(0, tsspec.mFullDataset.shape[0])
    #    tsspec.mFullDataset.fillna(method = "ffill", inplace = True);
    tsspec.mHorizon = {};
    for sig in tsspec.mFullDataset.columns:
        tsspec.mHorizon[sig] = 18
    #df_test = tsspec.mFullDataset.tail(tsspec.mHorizon);
    df_train = tsspec.mFullDataset;
    #.head(tsspec.mFullDataset.shape[0] - tsspec.mHorizon);
    tsspec.mTimeVar = "Date";
    tsspec.mSignalVar = "Signal";
    return tsspec;

# @profile    
def load_NN3_part2():
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "NN3_Part2";
    tsspec.mDescription = "NN3 competition final dataset - part 2";

    trainfile = "data/NN3-Final-Dataset-part2.csv"
    tsspec.mFullDataset = pd.read_csv(trainfile, sep='\t', header=0, engine='python');
    tsspec.mFullDataset['Date'] = np.arange(0, tsspec.mFullDataset.shape[0])
#    tsspec.mFullDataset.fillna(method = "ffill", inplace = True);
    #df_test = tsspec.mFullDataset.tail(tsspec.mHorizon);
    df_train = tsspec.mFullDataset
    tsspec.mHorizon = {};
    for sig in tsspec.mFullDataset.columns:
        tsspec.mHorizon[sig] = 18
    #.head(tsspec.mFullDataset.shape[0] - tsspec.mHorizon);
    tsspec.mTimeVar = "Date";
    tsspec.mSignalVar = "Signal";
    return tsspec;


# @profile    
def load_MWH_dataset(name):
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "MWH " + name;
    tsspec.mDescription = "MWH dataset ... " + name;

    lSignal = name;
    lTime = 'Time';
    print("LAODING_MWH_DATASET" , name);
    trainfile = "https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/FMA/" + name +".csv";
    # trainfile = "data/FMA/" + name + ".csv"
    df_train = pd.read_csv(trainfile, sep=r',', header=None,  engine='python', skipinitialspace=True);
    # print(df_train.head(3));
    type1 = np.dtype(df_train[df_train.columns[0]])
    if(type1.kind == 'O'):
        # there is (probably) a header, re-read the csv file
        df_train = pd.read_csv(trainfile, sep=r',', header=0,  engine='python', skipinitialspace=True);

    if(df_train.shape[1] == 1):
        # add dome fake date column
        df_train2 = pd.DataFrame();
        df_train2[lTime] = range(0, df_train.shape[0]);
        df_train2[lSignal] = df_train[df_train.columns[0]];
        df_train = df_train2.copy();
    # keep only the first two columns (as date and signal)
    df_train = df_train[[df_train.columns[0] , df_train.columns[1]]].dropna();
    # rename the first two columns (as date and signal)
    df_train.columns = [lTime , lSignal];
    # print("MWH_SIGNAL_DTYPE", df_train[lSignal].dtype)
    # print(df_train.head())
    # df_train.to_csv("mwh-" + name + ".csv");
    # print(df_train.info())
    if(df_train[lSignal].dtype == np.object):
        df_train[lSignal] = df_train[lSignal].astype(np.float64); ## apply(lambda x : float(str(x).replace(" ", "")));
    
    # df_train[lSignal] = df_train[lSignal].apply(float)

    tsspec.mFullDataset = df_train;
    # print(tsspec.mFullDataset.info())
    tsspec.mTimeVar = lTime;
    tsspec.mSignalVar = lSignal;
    tsspec.mHorizon = {};
    lHorizon = 1
    tsspec.mHorizon[lSignal] = lHorizon
    tsspec.mPastData = df_train[:-lHorizon];
    tsspec.mFutureData = df_train.tail(lHorizon);
    
    return tsspec


# @profile    
def load_M1_comp() :
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "M1_COMP";
    trainfile = "https://raw.githubusercontent.com/antoinecarme/pyaf/master/data/M1.csv"

    tsspec.mFullDataset = pd.read_csv(trainfile, sep='\t', header=0, engine='python');
    tsspec.mHorizons = tsspec.mFullDataset[['NF' , 'Series']].copy();
    tsspec.mHorizons['Series'] = tsspec.mHorizons['Series'].apply(lambda x  : x.replace(" ", ""))
    tsspec.mFullDataset['Index'] = tsspec.mFullDataset['Series']
    tsspec.mFullDataset['Index'] = tsspec.mFullDataset['Index'].apply(lambda x  : x.replace(" ", ""))
    tsspec.mFullDataset.drop(['Series', 'N Obs', 'Seasonality', 'NF', 'Type', 'Starting date', 'Category'], axis=1, inplace=True);
    tsspec.mFullDataset.set_index(['Index'], inplace=True)
    tsspec.mFullDataset = tsspec.mFullDataset.T
    tsspec.mFullDataset.reindex()
    tsspec.mFullDataset['Date'] = range(0 , tsspec.mFullDataset.shape[0])
    tsspec.mTimeVar = 'Date';

    tsspec.mHorizon = {};
    for i in range(0, tsspec.mHorizons.shape[0]):
        tsspec.mHorizon[tsspec.mHorizons['Series'][i]] = tsspec.mHorizons['NF'][i];
    del tsspec.mHorizons;
    
    return tsspec


# @profile    
def load_M2_comp() :
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "M2_COMP";
    trainfile = "https://raw.githubusercontent.com/antoinecarme/pyaf/master/data/M2.csv"

    tsspec.mFullDataset = pd.read_csv(trainfile, sep='\t', header=0, engine='python');
    tsspec.mHorizons = tsspec.mFullDataset[['NF' , 'Series']].copy();
    tsspec.mHorizons['Series'] = tsspec.mHorizons['Series'].apply(lambda x  : x.replace(" ", ""))
    tsspec.mFullDataset['Index'] = tsspec.mFullDataset['Series']
    tsspec.mFullDataset['Index'] = tsspec.mFullDataset['Index'].apply(lambda x  : x.replace(" ", ""))
    tsspec.mFullDataset.drop(['Series', 'N', 'Seasonality', 'Starting Date', 'Ending Date'], axis=1, inplace=True);
    tsspec.mFullDataset.set_index(['Index'], inplace=True)

    tsspec.mFullDataset = tsspec.mFullDataset.T
    tsspec.mFullDataset.reindex()
    tsspec.mFullDataset['Date'] = range(0 , tsspec.mFullDataset.shape[0])
    tsspec.mTimeVar = 'Date';

    tsspec.mHorizon = {};
    for i in range(0, tsspec.mHorizons.shape[0]):
        tsspec.mHorizon[tsspec.mHorizons['Series'][i]] = tsspec.mHorizons['NF'][i];
    #del tsspec.mHorizons;

    return tsspec


# @profile    
def load_M3_Y_comp() :
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "M3_Y_COMP";
    trainfile = "https://raw.githubusercontent.com/antoinecarme/pyaf/master/data/M3_Yearly.csv"

    tsspec.mFullDataset = pd.read_csv(trainfile, sep='\t', header=0, engine='python');
    tsspec.mHorizons = tsspec.mFullDataset[['NF' , 'Series']].copy();
    tsspec.mHorizons['Series'] = tsspec.mHorizons['Series'].apply(lambda x  : x.replace(" ", ""))
    tsspec.mFullDataset['Index'] = tsspec.mFullDataset['Series']
    tsspec.mFullDataset['Index'] = tsspec.mFullDataset['Index'].apply(lambda x  : x.replace(" ", ""))
    tsspec.mFullDataset.drop(['Series', 'N', 'NF', 'Starting Year' , 'Category' , 'Unnamed: 5'], axis=1, inplace=True);
    tsspec.mFullDataset.set_index(['Index'], inplace=True)

    tsspec.mFullDataset = tsspec.mFullDataset.T
    tsspec.mFullDataset.reindex()
    tsspec.mFullDataset['Date'] = range(0 , tsspec.mFullDataset.shape[0])
    tsspec.mTimeVar = 'Date';

    tsspec.mHorizon = {};
    for i in range(0, tsspec.mHorizons.shape[0]):
        tsspec.mHorizon[tsspec.mHorizons['Series'][i]] = tsspec.mHorizons['NF'][i];
    #del tsspec.mHorizons;

    return tsspec



# @profile    
def load_M3_Q_comp() :
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "M3_Q_COMP";
    trainfile = "https://raw.githubusercontent.com/antoinecarme/pyaf/master/data/M3_Quarterly.csv"

    tsspec.mFullDataset = pd.read_csv(trainfile, sep='\t', header=0, engine='python');
    tsspec.mHorizons = tsspec.mFullDataset[['NF' , 'Series']].copy();
    tsspec.mHorizons['Series'] = tsspec.mHorizons['Series'].apply(lambda x  : x.replace(" ", ""))
    tsspec.mFullDataset['Index'] = tsspec.mFullDataset['Series']
    tsspec.mFullDataset['Index'] = tsspec.mFullDataset['Index'].apply(lambda x  : x.replace(" ", ""))
    tsspec.mFullDataset.drop(['Series', 'N', 'NF', 'Starting Year' , 'Category' , 'Starting Quarter'], axis=1, inplace=True);
    tsspec.mFullDataset.set_index(['Index'], inplace=True)

    tsspec.mFullDataset = tsspec.mFullDataset.T
    tsspec.mFullDataset.reindex()
    tsspec.mFullDataset['Date'] = range(0 , tsspec.mFullDataset.shape[0])
    tsspec.mTimeVar = 'Date';

    tsspec.mHorizon = {};
    for i in range(0, tsspec.mHorizons.shape[0]):
        tsspec.mHorizon[tsspec.mHorizons['Series'][i]] = tsspec.mHorizons['NF'][i];
    #del tsspec.mHorizons;

    return tsspec


# @profile    
def load_M3_M_comp() :
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "M3_M_COMP";
    trainfile = "https://raw.githubusercontent.com/antoinecarme/pyaf/master/data/M3_Monthly.csv"

    tsspec.mFullDataset = pd.read_csv(trainfile, sep='\t', header=0, engine='python');
    tsspec.mHorizons = tsspec.mFullDataset[['NF' , 'Series']].copy();
    tsspec.mHorizons['Series'] = tsspec.mHorizons['Series'].apply(lambda x  : x.replace(" ", ""))
    tsspec.mFullDataset['Index'] = tsspec.mFullDataset['Series']
    tsspec.mFullDataset['Index'] = tsspec.mFullDataset['Index'].apply(lambda x  : x.replace(" ", ""))
    tsspec.mFullDataset.drop(['Series', 'N', 'NF', 'Starting Year' , 'Category' , 'Starting Month'], axis=1, inplace=True);
    tsspec.mFullDataset.set_index(['Index'], inplace=True)

    tsspec.mFullDataset = tsspec.mFullDataset.T
    tsspec.mFullDataset.reindex()
    tsspec.mFullDataset['Date'] = range(0 , tsspec.mFullDataset.shape[0])
    tsspec.mTimeVar = 'Date';

    tsspec.mHorizon = {};
    for i in range(0, tsspec.mHorizons.shape[0]):
        tsspec.mHorizon[tsspec.mHorizons['Series'][i]] = tsspec.mHorizons['NF'][i];
    #del tsspec.mHorizons;

    return tsspec




# @profile    
def load_M3_Other_comp() :
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "M3_Other_COMP";
    trainfile = "https://raw.githubusercontent.com/antoinecarme/pyaf/master/data/M3_Other.csv"

    tsspec.mFullDataset = pd.read_csv(trainfile, sep='\t', header=0, engine='python');
    tsspec.mHorizons = tsspec.mFullDataset[['NF' , 'Series']].copy();
    tsspec.mHorizons['Series'] = tsspec.mHorizons['Series'].apply(lambda x  : x.replace(" ", ""))
    tsspec.mFullDataset['Index'] = tsspec.mFullDataset['Series']
    tsspec.mFullDataset['Index'] = tsspec.mFullDataset['Index'].apply(lambda x  : x.replace(" ", ""))
    tsspec.mFullDataset.drop(['Series', 'N', 'NF', 'Category', 'Unnamed: 4', 'Unnamed: 5'], axis=1, inplace=True);
    tsspec.mFullDataset.set_index(['Index'], inplace=True)

    tsspec.mFullDataset = tsspec.mFullDataset.T
    tsspec.mFullDataset.reindex()
    tsspec.mFullDataset['Date'] = range(0 , tsspec.mFullDataset.shape[0])
    tsspec.mTimeVar = 'Date';

    tsspec.mHorizon = {};
    for i in range(0, tsspec.mHorizons.shape[0]):
        tsspec.mHorizon[tsspec.mHorizons['Series'][i]] = tsspec.mHorizons['NF'][i];
    #del tsspec.mHorizons;

    return tsspec

# @profile    
def load_M4_comp(iType = None) :
    """
    generated by script data/m4comp.R using the excellent M4Comp R package.
    """

    tsspecs = {};
    
    trainfile = "https://github.com/antoinecarme/pyaf/blob/master/data/M4Comp/M4Comp_" + iType + ".csv.gz?raw=true"
    # trainfile = "data/M4Comp/M4Comp_" + iType + ".csv.gz"

    df_full = pd.read_csv(trainfile, sep=',', header=0, engine='python', compression='gzip');
    lHorizons = df_full[['H' , 'ID']].copy();
    lHorizons['ID'] = lHorizons['ID'].apply(lambda x  : x.replace(" ", ""));
    lHorizons['H'] = lHorizons['H'].apply(lambda x  : int(x))

    for i in range(0, df_full.shape[0]):
        tsspec = cTimeSeriesDatasetSpec();
        series_name = lHorizons['ID'][i]
        tsspec.mName = series_name;

        if(((i+1) % 500) == 0):
            print("loading ", i+1 , "/" , df_full.shape[0] , series_name);
        tsspec.mPastData = pd.Series(df_full['PAST'][i].split(",")).apply(float);
        tsspec.mFutureData = pd.Series(df_full['FUTURE'][i].split(",")).apply(float);
        tsspec.mFullDataset = pd.DataFrame();
        tsspec.mFullDataset[series_name] = tsspec.mPastData.append(tsspec.mFutureData).reindex();
        tsspec.mFullDataset['Date'] = range(0 , tsspec.mFullDataset.shape[0])
        tsspec.mTimeVar = "Date";
        tsspec.mSignalVar = series_name;
        tsspec.mFullDataset.reindex()
        tsspec.mHorizon = {};
        tsspec.mHorizon[series_name] = lHorizons['H'][i];
        tsspec.mCategory = "M4Comp";
        tsspecs[tsspec.mName] = tsspec;

    return tsspecs

def get_stock_web_link():
    YAHOO_LINKS_DATA = {}
    lines = [line.rstrip('\n') for line in open('data/yahoo_list.txt')]
    import re
    for line in lines:
        csv = line.replace('.csv', '')
        csv = re.sub(r"^(.*)yahoo_", "", csv);
        # print("YAHOO_LINKS_DATA" , csv, line)
        YAHOO_LINKS_DATA[csv] = line;
    print("ACQUIRED_YAHOO_LINKS" , len(YAHOO_LINKS_DATA));
    return YAHOO_LINKS_DATA;

def load_yahoo_stock_price( stock , iLocal = True, YAHOO_LINKS_DATA = None) :
    filename = YAHOO_LINKS_DATA.get(stock);
    if(filename is None):
        raise Exception("MISSING " + stock)
        
    # print("YAHOO_DATA_LINK" , stock, filename);

    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "Yahoo_Stock_Price_" + stock 
    tsspec.mDescription = "Yahoo Stock Price using yahoo-finance package"
    df_train = pd.DataFrame();
    if(iLocal):
        filename = "data/yahoo/" + filename
    else:
        base_uri = "https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/YahooFinance/";
        filename =  base_uri + filename;
    print("YAHOO_DATA_LINK_URI" , stock, filename);
    if(os.path.isfile(filename)):
        # print("already downloaded " + stock , "reloading " , filename);
        df_train = pd.read_csv(filename);
    else:
        # return None;
        from yahoo_finance import Share
        stock_obj = Share(stock)
        today = date.today()
        today
        before = date(today.year - 5, today.month, today.day)
        # print(today, before)
        lst = stock_obj.get_historical(before.isoformat(), today.isoformat())
        # print(stock , len(lst));
        if(len(lst) > 0):
            for k in lst[0].keys():
                for i in range(0, len(lst)):
                    lst_k = [];
                    for line1 in lst:
                        lst_k = lst_k + [line1[k]];
                df_train[k] = lst_k;
            # df_train.to_csv(filename);
        else:
            # df_train.to_csv(filename);
            return None            

    tsspec.mFullDataset = pd.DataFrame();
    tsspec.mFullDataset[stock] = df_train['Close'].apply(float);
    tsspec.mFullDataset['Date'] = df_train['Date'].apply(lambda x : datetime.datetime.strptime(x, "%Y-%m-%d"))
#    print(tsspec.mFullDataset.head());
    tsspec.mFullDataset.sort_values(by = 'Date' , ascending=True, inplace=True)
    tsspec.mFullDataset.reset_index(inplace = True, drop=True);
#    print(tsspec.mFullDataset.head());
    tsspec.mTimeVar = "Date";
    tsspec.mSignalVar = stock;
    lHorizon = 7 # 7 days
    if(lHorizon > tsspec.mFullDataset.shape[0]):
        # nysecomp/yahoo_VRS.csv is too small
        lHorizon = 1
    tsspec.mHorizon = {};
    tsspec.mHorizon[stock] = lHorizon
    tsspec.mPastData = tsspec.mFullDataset[:-lHorizon];
    tsspec.mFutureData = tsspec.mFullDataset.tail(lHorizon);

    # print(tsspec.mFullDataset.head())
    return tsspec    


# @profile    
def load_yahoo_stock_prices(symbol_list_key) :
    tsspecs = {}

    YAHOO_LINKS_DATA = get_stock_web_link();

    stocks = symlist.SYMBOL_LIST[symbol_list_key]
    for stock in sorted(stocks):
        tsspec1 = None
        try:
            tsspec1 = load_yahoo_stock_price(stock , True, YAHOO_LINKS_DATA)
        except:
            # raise
            pass
        
        if(tsspec1 is not None) :
            tsspec1.mCategory = symbol_list_key;
            tsspecs[stock] = tsspec1; 

    print("load_yahoo_stock_prices" , symbol_list_key, len(tsspecs.keys()));
    return tsspecs



class cYahoo_download_Arg_Arg:
    def __init__(self , stocks):
        self.mList = stocks;

def download_Yahoo_list(arg):
    YAHOO_LINKS_DATA = get_stock_web_link();
    for k in arg.mList:
        try:
            load_yahoo_stock_price(k, False, YAHOO_LINKS_DATA)
        except:
            pass

def download_yahoo_stock_prices() :
    import multiprocessing as mp
    pool = mp.Pool()
    args = []
    for k in symlist.SYMBOL_LIST.keys():
        lst = symlist.SYMBOL_LIST[k]
        n = 0;
        N = len(lst);
        while(n <= N):
            end1 = n + 50
            if(end1 > N):
                end1 = N;
            lst1 = lst[n : end1]
            arg = cYahoo_download_Arg_Arg(lst1)
            args = args + [arg];
            n = n + 50;
                
    asyncResult = pool.map_async(download_Yahoo_list, args);

    resultList = asyncResult.get()


def get_yahoo_symbol_lists():
    return symlist.SYMBOL_LIST;




# @profile    
def generate_datasets(ds_type = "S"):
    datasets = {};
    lRange_N = range(20, 101, 20)
    if(ds_type == "M"):
        lRange_N = range(150, 501, 50)
    if(ds_type == "L"):
        lRange_N = range(600, 2001, 100)
    if(ds_type == "XL"):
        lRange_N = range(2500, 8000, 500)
    
    for N in lRange_N:
        for trend in ["constant" , "linear" , "poly"]:
            for cycle_length in range(0, N // 4 ,  max(N // 16 , 1)):
                for transf in ["" , "exp"]:            
                    for sigma in range(0, 5, 2):
                        for exogc in range(0, 51, 20):
                            for seed in range(0, 1):
                                ds = generate_random_TS(N , 'D', seed, trend,
                                                        cycle_length, transf,
                                                        sigma, exog_count = exogc);
                                ds.mCategory = "ARTIFICIAL_" + ds_type;
                                datasets[ds.mName] = ds
    return datasets;


# @profile    
def load_artificial_datsets(ds_type = "S") :

    tsspecs = generate_datasets(ds_type);
    print("ARTIFICIAL_DATASETS_TESTED" , len(tsspecs))

    return tsspecs

def load_MWH_datsets() :
    datasets = "10-6 11-2 9-10 9-11 9-12 9-13 9-17a 9-17b 9-1 9-2 9-3 9-4 9-5 9-9 advert adv_sale airline bankdata beer2 bicoal books boston bricksq canadian capital cars cement computer condmilk cow cpi_mel deaths dexter dj dole dowjones eknives elco elec2 elec elecnew ex2_6 ex5_2 expendit fancy french fsales gas housing hsales2 hsales huron ibm2 input invent15 jcars kkong labour lynx milk mink mortal motel motion olympic ozone paris pcv petrol pigs plastics pollutn prodc pulppric qsales res running sales schizo shampoo sheep ship shipex strikes temperat ukdeaths ustreas wagesuk wn wnoise writing".split(" ");

    # datasets = "milk".split(" ");

    tsspecs = {};
    for ds in datasets:
        if(ds != "cars"):
            tsspecs[ds] = load_MWH_dataset(ds);
            tsspecs[ds].mCategory = "MWH"; 

    return tsspecs;


# @profile    
def load_AU_hierarchical_dataset():
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "Ozone"
    tsspec.mDescription = "https://www.otexts.org/fpp/9/4"

    trainfile = "data/Hierarchical/hts_dataset.csv";
    lDateColumn = 'Date'

    df_train = pd.read_csv(trainfile, sep=r',', engine='python', skiprows=0);
    df_train[lDateColumn] = df_train[lDateColumn].apply(lambda x : datetime.datetime.strptime(x, "%Y-%m-%d"))
    
    tsspec.mTimeVar = lDateColumn;
    tsspec.mSignalVar = None;
    tsspec.mHorizon = 12;
    tsspec.mPastData = df_train[:-tsspec.mHorizon];
    tsspec.mFutureData = df_train.tail(tsspec.mHorizon);

    rows_list = [];
    # Sydney    NSW  Melbourne    VIC  BrisbaneGC    QLD  Capitals Other
    rows_list.append(['Sydney' , 'NSW_State' , 'Australia']);
    rows_list.append(['NSW' , 'NSW_State' , 'Australia']);
    rows_list.append(['Melbourne' , 'VIC_State' , 'Australia']);
    rows_list.append(['VIC' , 'VIC_State' , 'Australia']);
    rows_list.append(['BrisbaneGC' , 'QLD_State' , 'Australia']);
    rows_list.append(['QLD' , 'QLD_State' , 'Australia']);
    rows_list.append(['Capitals' , 'Other_State' , 'Australia']);
    rows_list.append(['Other' , 'Other_State' , 'Australia']);

    lLevels = ['City' , 'State' , 'Country'];
    lHierarchy = {};
    lHierarchy['Levels'] = lLevels;
    lHierarchy['Data'] = pd.DataFrame(rows_list, columns =  lLevels);
    lHierarchy['Type'] = "Hierarchical";
    
    print(lHierarchy['Data'].head(lHierarchy['Data'].shape[0]));

    tsspec.mHierarchy = lHierarchy;
    
    return tsspec



# @profile    
def load_AU_infant_grouped_dataset():
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "Ozone"
    tsspec.mDescription = "https://cran.r-project.org/web/packages/hts/hts.pdf";
    
    trainfile = "data/Hierarchical/infant_gts.csv";
    lDateColumn = 'Index'

    df_train = pd.read_csv(trainfile, sep=r',', engine='python', skiprows=0);
    # df_train[lDateColumn] = df_train[lDateColumn].apply(lambda x : datetime.datetime.strptime(x, "%Y-%m-%d"))
    
    tsspec.mTimeVar = lDateColumn;
    tsspec.mSignalVar = None;
    tsspec.mHorizon = 12;
    tsspec.mPastData = df_train[:-tsspec.mHorizon];
    tsspec.mFutureData = df_train.tail(tsspec.mHorizon);

    lGroups = {};
    lGroups["State"] = ["NSW","VIC","QLD","SA","WA","NT","ACT","TAS"];
    lGroups["Gender"] = ["female","male"];
    # lGroups["Gender1"] = ["femme","homme"];
    # lGroups["age"] = ["1", "2","3"];
    lHierarchy = {};
    lHierarchy['Levels'] = None;
    lHierarchy['Data'] = None;
    lHierarchy['Groups']= lGroups;
    lHierarchy['GroupOrder']= ["State" , "Gender"]; # by state first, then by gender
    lHierarchy['Type'] = "Grouped";
    
    tsspec.mHierarchy = lHierarchy;
    
    return tsspec


def load_fpp2_dataset(name):
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "FPP2";
    tsspec.mDescription = "https://github.com/robjhyndman/fpp2-package ... " + name;

    lSignal = name;
    lTime = 'Time';
    # trainfile = "/home/antoine/dev/python/packages/TimeSeriesData/fpp2/" + name +".csv";
    trainfile = "https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/fpp2/" + name +".csv";
    df_train = pd.read_csv(trainfile, sep=r',',  engine='python', skipinitialspace=True);
    print("LAODING_FPP2_DATASET", name , list(df_train.columns))

    if(df_train.shape[1] == 1):
        # add dome fake date column
        df_train2 = pd.DataFrame();
        df_train2[lTime] = range(0, df_train.shape[0]);
        df_train2[lSignal] = df_train[df_train.columns[0]];
        df_train = df_train2.copy();
    # keep only the first two columns (as date and signal)
    df_train = df_train[[df_train.columns[0] , df_train.columns[1]]].dropna();
    # rename the first two columns (as date and signal)
    df_train.columns = [lTime , lSignal];
    if(df_train[lSignal].dtype == np.object):
        df_train[lSignal] = df_train[lSignal].astype(np.float64); 

    print(df_train.head(5));
    # df_train.info()

    tsspec.mFullDataset = df_train;
    # print(tsspec.mFullDataset.info())
    tsspec.mTimeVar = lTime;
    tsspec.mSignalVar = lSignal;
    tsspec.mHorizon = {};
    lHorizon = 4
    tsspec.mHorizon[lSignal] = lHorizon
    tsspec.mPastData = df_train[:-lHorizon];
    tsspec.mFutureData = df_train.tail(lHorizon);
    
    return tsspec



def load_FPP2_datsets() :
    fpp2_datasets = ["goog200", "auscafe", "sunspotarea", "elecdemand", "h02", "ausair", "usmelec", "uschange", "qgas", "ausbeer", "livestock", "mens400", "elecsales", "arrivals", "prison", "wmurders", "departures", "qauselec", "goog", "visnights", "hyndsight", "prisonLF", "a10", "debitcards", "melsyd", "marathon", "elecdaily", "insurance", "oil", "maxtemp", "calls", "guinearice", "qcement", "elecequip", "austourists", "gasoline", "austa", "euretail"]

    tsspecs = {};
    for ds in fpp2_datasets:
        if(ds != "prisonLF"):
            tsspecs[ds] = load_fpp2_dataset(ds);
            tsspecs[ds].mCategory = "FPP2"; 

    return tsspecs;
