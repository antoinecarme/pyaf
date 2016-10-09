
import pandas as pd
import numpy as np
import datetime
from datetime import date
from yahoo_finance import Share
import os.path
import AutoForecast.data.stocks_symbol_list as symlist

import multiprocessing as mp
import threading
from multiprocessing.dummy import Pool as ThreadPool

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


    
def load_cashflows() :
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "CashFlows"
    tsspec.mDescription = "CashFlows dataset"

    trainfile = "data/CashFlows.txt"
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

def load_ozone() :
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "Ozone"
    tsspec.mDescription = "https://datamarket.com/data/set/22u8/ozon-concentration-downtown-l-a-1955-1972"
    
    #trainfile = "data/ozone-la.csv"
    trainfile = "https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/ozone-la.csv"

    cols = ["Month" , "Ozone"];
    
    df_train = pd.read_csv(trainfile, names = cols, sep=r',', engine='python', skiprows=1);
    df_train['Time'] = df_train['Month'].apply(lambda x : datetime.datetime.strptime(x, "%Y-%m"))

    tsspec.mTimeVar = "Time";
    tsspec.mSignalVar = "Ozone";
    tsspec.mHorizon = 12;
    tsspec.mPastData = df_train[:-tsspec.mHorizon];
    tsspec.mFutureData = df_train.tail(tsspec.mHorizon);
    
    return tsspec

def load_ozone_exogenous() :
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "Ozone"
    tsspec.mDescription = "https://datamarket.com/data/set/22u8/ozon-concentration-downtown-l-a-1955-1972"
    
    #trainfile = "data/ozone-la.csv"
    trainfile = "data/ozone-la-exogenous.csv"
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
    
    return tsspec

def add_some_noise(x , p):
    if(np.random.random() < p):
        return "A" + str(int(0.01*x));
    return "";

def generate_random_TS(N , FREQ, seed, trendtype, cycle_length, transform, sigma = 1.0, exog_count = 20) :
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

    if(trendtype == "constant"):
        df_train['GeneratedTrend'] = N *  (2 * np.random.random() - 1)
    if(trendtype == "linear"):
        x = np.arange(0,N) / (N + 0.1);
        df_train['GeneratedTrend'] =  N * ((2 * np.random.random() - 1) -  (2 * np.random.random() - 1) * x)
    if(trendtype == "poly"):
        x = np.arange(0, N);
        df_train['GeneratedTrend'] =  (2 * np.random.random() - 1) + (2 * np.random.random() - 1) * x
        df_train['GeneratedTrend'] +=  (2 * np.random.random() - 1) * x ** 2
        df_train['GeneratedTrend'] = N * df_train['GeneratedTrend'];

    if(cycle_length > 0):
        df_train['GeneratedCycle'] = np.arange(0,N) % cycle_length;
        lValues = np.random.randint(0, cycle_length, size=(cycle_length, 1)) * (N + 0.0)/cycle_length;
        df_train['GeneratedCycle'] = df_train['GeneratedCycle'].apply(lambda x : lValues[int(x)][0]);
    if(cycle_length == 0):
        df_train['GeneratedCycle'] = 0;

    df_train['Noise'] = np.random.randn(N, 1) * sigma;
    df_train['Signal'] = df_train['GeneratedTrend'] +  df_train['GeneratedCycle'] + df_train['Noise']

    tsspec.mExogenousVariables = [];
    for e in range(exog_count):
        label = "exog_" + str(e+1);
        df_train[label] = df_train['Signal'].apply(lambda x : add_some_noise(x , 0.1));
        tsspec.mExogenousVariables = tsspec.mExogenousVariables + [ label ];

    # this is the full dataset . must contain future exogenius data
    tsspec.mExogenousDataFrame = df_train;
    min_sig = df_train['Signal'].min();
    pos_signal = df_train['Signal'] - min_sig + 1.0;

    if(transform == "exp"):
        df_train['Signal'] = np.exp(-pos_signal)

    # df_train.to_csv(tsspec.mName + ".csv");

    tsspec.mTimeVar = "Date";
    tsspec.mSignalVar = "Signal";
    tsspec.mHorizon = 12;
    if(tsspec.mHorizon > (N//2)):
        tsspec.mHorizon = N // 2;
    tsspec.mFullDataset = df_train;
    tsspec.mFullDataset[tsspec.mName] = tsspec.mFullDataset['Signal'];
    tsspec.mPastData = df_train[:-tsspec.mHorizon];
    tsspec.mFutureData = df_train.tail(tsspec.mHorizon);
    
    return tsspec


def load_NN5():
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "NN5";
    tsspec.mDescription = "NN5 competition final dataset";
    trainfile = "data/NN5-Final-Dataset.csv"
    tsspec.mFullDataset = pd.read_csv(trainfile, sep='\t', header=0, engine='python');
    tsspec.mFullDataset['Date'] = tsspec.mFullDataset['Day'].apply(lambda x : datetime.datetime.strptime(x, "%d-%b-%y"))
    tsspec.mFullDataset = tsspec.mFullDataset.drop(['Day']  , axis=1)
    tsspec.mFullDataset.fillna(method = "ffill", inplace = True);
    tsspec.mHorizon = 56
#    df_test = tsspec.mFullDataset.tail(tsspec.mHorizon);
    df_train = tsspec.mFullDataset;

    tsspec.mTimeVar = "Day";
    tsspec.mSignalVar = "Signal";
    tsspec.mHorizon = 12;
    tsspec.mPastData = df_train[:-tsspec.mHorizon];
    tsspec.mFutureData = df_train.tail(tsspec.mHorizon);
    
    return tsspec


def load_NN3_part1():
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "NN3_Part1";
    tsspec.mDescription = "NN3 competition final dataset - part 1";
    trainfile = "data/NN3-Final-Dataset-part1.csv"
    tsspec.mFullDataset = pd.read_csv(trainfile, sep='\t', header=0, engine='python');
    tsspec.mFullDataset['Date'] = np.arange(0, tsspec.mFullDataset.shape[0])
#    tsspec.mFullDataset.fillna(method = "ffill", inplace = True);
    tsspec.mHorizon = 18
    #df_test = tsspec.mFullDataset.tail(tsspec.mHorizon);
    df_train = tsspec.mFullDataset;
    #.head(tsspec.mFullDataset.shape[0] - tsspec.mHorizon);
    return tsspec;

def load_NN3_part2():
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "NN3_Part2";
    tsspec.mDescription = "NN3 competition final dataset - part 2";

    trainfile = "data/NN3-Final-Dataset-part2.csv"
    tsspec.mFullDataset = pd.read_csv(trainfile, sep='\t', header=0, engine='python');
    tsspec.mFullDataset['Date'] = np.arange(0, tsspec.mFullDataset.shape[0])
#    tsspec.mFullDataset.fillna(method = "ffill", inplace = True);
    tsspec.mHorizon = 18
    #df_test = tsspec.mFullDataset.tail(tsspec.mHorizon);
    df_train = tsspec.mFullDataset
    #.head(tsspec.mFullDataset.shape[0] - tsspec.mHorizon);
    return tsspec;


def load_MWH_dataset(name):
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "MWH " + name;
    tsspec.mDescription = "MWH dataset ... " + name;

    lSignal = name;
    lTime = 'Time';
    
    trainfile = "data/FMA/" + name + ".csv"
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
    tsspec.mHorizon = 1;
    tsspec.mPastData = df_train[:-tsspec.mHorizon];
    tsspec.mFutureData = df_train.tail(tsspec.mHorizon);
    
    return tsspec


def load_M1_comp() :
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "M1_COMP";
    trainfile = "data/M1.csv"

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

    tsspec.mHorizon = {};
    for i in range(0, tsspec.mHorizons.shape[0]):
        tsspec.mHorizon[tsspec.mHorizons['Series'][i]] = tsspec.mHorizons['NF'][i];
    del tsspec.mHorizons;
    
    return tsspec


def load_M2_comp() :
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "M2_COMP";
    trainfile = "data/M2.csv"

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

    tsspec.mHorizon = {};
    for i in range(0, tsspec.mHorizons.shape[0]):
        tsspec.mHorizon[tsspec.mHorizons['Series'][i]] = tsspec.mHorizons['NF'][i];
    #del tsspec.mHorizons;

    return tsspec


def load_M3_Y_comp() :
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "M3_Y_COMP";
    trainfile = "data/M3_Yearly.csv"

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

    tsspec.mHorizon = {};
    for i in range(0, tsspec.mHorizons.shape[0]):
        tsspec.mHorizon[tsspec.mHorizons['Series'][i]] = tsspec.mHorizons['NF'][i];
    #del tsspec.mHorizons;

    return tsspec



def load_M3_Q_comp() :
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "M3_Q_COMP";
    trainfile = "data/M3_Quarterly.csv"

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

    tsspec.mHorizon = {};
    for i in range(0, tsspec.mHorizons.shape[0]):
        tsspec.mHorizon[tsspec.mHorizons['Series'][i]] = tsspec.mHorizons['NF'][i];
    #del tsspec.mHorizons;

    return tsspec


def load_M3_M_comp() :
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "M3_M_COMP";
    trainfile = "data/M3_Monthly.csv"

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

    tsspec.mHorizon = {};
    for i in range(0, tsspec.mHorizons.shape[0]):
        tsspec.mHorizon[tsspec.mHorizons['Series'][i]] = tsspec.mHorizons['NF'][i];
    #del tsspec.mHorizons;

    return tsspec




def load_M3_Other_comp() :
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "M3_Other_COMP";
    trainfile = "data/M3_Other.csv"

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

    tsspec.mHorizon = {};
    for i in range(0, tsspec.mHorizons.shape[0]):
        tsspec.mHorizon[tsspec.mHorizons['Series'][i]] = tsspec.mHorizons['NF'][i];
    #del tsspec.mHorizons;

    return tsspec

def load_M4_comp() :
    """
    generated by script data/m4comp.R using the excellent M4Comp R package.
    """

    tsspecs = {};
    
    trainfile = "data/M4Comp.csv"

    df_full = pd.read_csv(trainfile, sep=',', header=0, engine='python');
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
        tsspec.mHorizon = lHorizons['H'][i];
        tsspec.mCategory = "M4Comp";
        tsspecs[tsspec.mName] = tsspec;

    return tsspecs


def load_yahoo_stock_price( stock ) :
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "Yahoo_Stock_Price_" + stock 
    tsspec.mDescription = "Yahoo Stock Price using yahoo-finance package"
    df_train = pd.DataFrame();
    filename = "data/yahoo/yahoo_" + stock +".csv"
    if(os.path.isfile(filename)):
        # print("already downloaded " + stock , "reloading " , filename);
        df_train = pd.read_csv(filename);
    else:
        return None;
        stock_obj = Share(stock)
        today = date.today()
        today
        before = date(today.year - 5, today.month, today.day)
        print(today, before)
        lst = stock_obj.get_historical(before.isoformat(), today.isoformat())
        print(stock , len(lst));
        if(len(lst) > 0):
            for k in lst[0].keys():
                for i in range(0, len(lst)):
                    lst_k = [];
                    for line1 in lst:
                        lst_k = lst_k + [line1[k]];
                df_train[k] = lst_k;
            df_train.to_csv(filename);
        else:
            # df_train.to_csv(filename);
            return None            

    tsspec.mFullDataset = pd.DataFrame();
    tsspec.mFullDataset[stock] = df_train['Close'] 
    tsspec.mFullDataset['Date'] = df_train['Date'].apply(lambda x : datetime.datetime.strptime(x, "%Y-%m-%d"))
#    print(tsspec.mFullDataset.head());
    tsspec.mFullDataset.sort_values(by = 'Date' , ascending=True, inplace=True)
    tsspec.mFullDataset.reset_index(inplace = True);
#    print(tsspec.mFullDataset.head());
    tsspec.mTimeVar = "Date";
    tsspec.mSignalVar = stock;
    tsspec.mHorizon = 7; # 7 days
    tsspec.mPastData = tsspec.mFullDataset[:-tsspec.mHorizon];
    tsspec.mFutureData = tsspec.mFullDataset.tail(tsspec.mHorizon);
    
    return tsspec    


def load_yahoo_stock_prices(symbol_list_key) :
    tsspecs = {}

    stocks = symlist.SYMBOL_LIST[symbol_list_key]
    for stock in stocks:
        try:
            tsspec1 = load_yahoo_stock_price(stock)
        except:
            tsspec1 = None
            pass
        
        if(tsspec1) :
            tsspec1.mCategory = symbol_list_key;
            tsspecs[stock] = tsspec1; 

    print("load_yahoo_stock_prices" , symbol_list_key, len(tsspecs.keys()));
    return tsspecs



class cYahoo_download_Arg_Arg:
    def __init__(self , stocks):
        self.mList = stocks;

def download_Yahoo_list(arg):
    for k in arg.mList:
        try:
            load_yahoo_stock_price(k)
        except:
            pass

def download_yahoo_stock_prices() :
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
    lHierarchy['GroupOrder']= ["State" , "Gender"];
    lHierarchy['Type'] = "Grouped";
    
    tsspec.mHierarchy = lHierarchy;
    
    return tsspec
