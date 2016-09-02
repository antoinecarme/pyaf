
import pandas as pd
import numpy as np
import datetime
from datetime import date
from yahoo_finance import Share
import os.path
import data.stocks_symbol_list as symlist

import multiprocessing as mp
import threading
from multiprocessing.dummy import Pool as ThreadPool

class cTimeSeriesDatasetSpec:

    def __init__(self):
        self.mSignalFrame = pd.DataFrame()

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


def generate_random_TS(N , FREQ, seed, trendtype, cycle_length, transform, sigma = 1.0) :
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "Random_Generated_Dataset_" + str(N) + "_" + str(FREQ);
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
        df_train['GeneratedTrend'] = 0.5
    if(trendtype == "linear"):
        x = np.arange(0,N)  * (10.0 / N);
        df_train['GeneratedTrend'] = 200.0 - 2.0 * x
    if(trendtype == "poly"):
        x = np.arange(0,N)  * (10.0 / N);
        df_train['GeneratedTrend'] = 200.0 - 2.0 * x
        df_train['GeneratedTrend'] += 4.0 * x ** 2

    if(cycle_length > 0):
        df_train['GeneratedCycle'] = np.arange(0,N) % cycle_length;
    if(cycle_length == 0):
        df_train['GeneratedCycle'] = 0;
        
    df_train['Noise'] = np.random.randint(0, N, size=(N, 1)) * sigma;
    df_train['Signal'] = df_train['GeneratedTrend'] +  df_train['GeneratedCycle'] + df_train['Noise']

    min_sig = df_train['Signal'].min();
    pos_signal = df_train['Signal'] - min_sig + 1.0;

    if(transform == "exp"):
        df_train['Signal'] = np.exp(-pos_signal)

#    df_train.to_csv(tsspec.mName + ".csv");

    tsspec.mTimeVar = "Date";
    tsspec.mSignalVar = "Signal";
    tsspec.mHorizon = 12;
    if(tsspec.mHorizon > (N//2)):
        tsspec.mHorizon = N // 2;
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
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "M4_COMP";
    trainfile = "data/M4Comp.csv"

    df_full = pd.read_csv(trainfile, sep=',', header=0, engine='python');
    tsspec.mHorizons = df_full[['H' , 'ID']].copy();
    tsspec.mHorizons['ID'] = tsspec.mHorizons['ID'].apply(lambda x  : x.replace(" ", ""));
    tsspec.mHorizons['H'] = tsspec.mHorizons['H'].apply(lambda x  : int(x))

    tsspec.mHorizon = {};
    for i in range(0, tsspec.mHorizons.shape[0]):
        tsspec.mHorizon[tsspec.mHorizons['ID'][i]] = tsspec.mHorizons['H'][i];

    lMaxLength = 0;
    for i in range(0, df_full.shape[0]):
        lMaxLength = max(lMaxLength, df_full['N'][i] +  df_full['H'][i]);
    print("MAX_LENGTH" , lMaxLength);
    tsspec.mFullDataset = pd.DataFrame();
    for i in range(0, df_full.shape[0]):
        series_name = df_full['ID'][i]
        if(((i+1) % 500) == 0):
            print("loading ", i+1 , "/" , df_full.shape[0] , series_name);
#        print("PAST", i, df_full['PAST'][i]);
        past = pd.Series(df_full['PAST'][i].split(","));
        past = past.apply(lambda x  : float(x));
#        print("PAST2", i, past);
        future = pd.Series(df_full['FUTURE'][i].split(","));
        future = future.apply(lambda x  : float(x));
#        future.reset_index(inplace=True);
        notPadded = pd.Series(past);
        notPadded = notPadded.append(future);
        notPadded.reindex();
        lpadding = pd.Series(np.zeros(lMaxLength - notPadded.shape[0]))
        lpadding = lpadding.apply(lambda x : np.nan);
#        print("Padding", i, series_name, past.shape[0], future.shape[0], lpadding.shape[0]);
        lPadded = notPadded.append(lpadding);
        #notPadded.reindex();
        lPadded.reset_index(drop = True, inplace=True);
        assert(lPadded.shape[0] == lMaxLength)
#        print("notPadded", i, notPadded);
        #lfinal.reset_index(inplace=True);
        #tsspec.mFullDataset.reindex();
        tsspec.mFullDataset[series_name] = lPadded;
#        print("SERIES", i, tsspec.mFullDataset[series_name]);
    
    tsspec.mFullDataset.reindex()
    tsspec.mFullDataset['Date'] = range(0 , tsspec.mFullDataset.shape[0])
    #del tsspec.mHorizons;

    return tsspec


def load_yahoo_stock_price( stock ) :
    tsspec = cTimeSeriesDatasetSpec();
    tsspec.mName = "Yahoo_Stock_Price_" + stock 
    tsspec.mDescription = "Yahoo Stock Price using yahoo-finance package"
    df_train = pd.DataFrame();
    filename = "data/yahoo/yahoo_" + stock +".csv"
    if(os.path.isfile(filename)):
        print("already downloaded " + stock , "reloading " , filename);
        df_train = pd.read_csv(filename);
    else:
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
    tsspec.mPastData = df_train[:-tsspec.mHorizon];
    tsspec.mFutureData = df_train.tail(tsspec.mHorizon);
    
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
