import numpy as np
import pandas as pd
from . import SignalDecomposition_AR as tsar
from . import Utils as tsutil

import sys


def is_signal_intermittent(iSeries, iOptions):
    series = iSeries - iSeries.min()
    zero_values = series[abs(series) < 1e-8]
    lZeroRate = zero_values.shape[0] / series.shape[0]
    if(lZeroRate > iOptions.mCrostonOptions.mZeroRate):
        return True
    return False
    

class cCroston_Model(tsar.cAbstractAR):
    def __init__(self , cycle_residue_name, P , iExogenousInfo = None):
        super().__init__(cycle_residue_name, iExogenousInfo)
        self.mNbLags = 1

    def dumpCoefficients(self, iMax=10):
        # print(self.mScikitModel.__dict__);
        pass

    def set_name(self):
        self.mOutName = self.mCycleResidueName +  '_CROSTON(' + str(self.mOptions.mCrostonOptions.mAlpha) + ')';
        self.mFormula = "CROSTON"

    def get_coeff(self, alpha , croston_type):
        if(croston_type == "SBA"):
            return 1.0-(alpha/2.0)
        elif(croston_type == "SBJ"):
            return (1.0 - alpha/(2.0-alpha))
        # default : any other value is the legacy croston method
        return 1.0

    def croston(self, df, horizon_index = 1):
        # print(df.shape)
        # print(df.columns)
        # print(df[['Date', 'Signal', '_Signal', 'row_number', '_Signal_ConstantTrend_residue_zeroCycle_residue']].tail(12))
        alpha =  self.mOptions.mCrostonOptions.mAlpha
        method = self.mOptions.mCrostonOptions.mMethod
        lCounts_df = df[[self.mTime, self.mCycleResidueName]].copy()
        counts = lCounts_df[self.mCycleResidueName] - self.mOffset
        counts = counts[:-(horizon_index)]
        # print(list(counts.unique()))
        # print(counts.describe())
        # assert(not np.isnan(counts[:-1]).any())
        #  q is often called the “demand” and a the “inter-arrival time”.
        q = counts[abs(counts) > 1e-8]
        demand_times = pd.Series(list(q.index)) + 1
        a = demand_times - demand_times.shift(1).fillna(0.0)
        df2 = pd.DataFrame({'demand_time' : list(demand_times), 'q' : list(q) , 'a' : list(a) })
        
        df2['q_est'] = None
        df2['a_est'] = None
        
        # initialization : first values
        df2.loc[0 , 'q_est'] = df2['q'][0]
        df2.loc[0,  'a_est'] = df2['a'][0]
        for i in range(df2.shape[0] - 1):
            q1 = (1.0 - alpha) * df2['q_est'][ i ] + alpha * df2['q'][ i ]
            a1 = (1.0 - alpha) * df2['a_est'][ i ] + alpha * df2['a'][ i ]
            df2.loc[i + 1, 'q_est'] = q1
            df2.loc[i + 1, 'a_est'] = a1

        df2['forecast'] = self.get_coeff(alpha , method) * df2['q_est'] / df2['a_est']
        df2['index'] = df2['demand_time'] - 1

        for h in range(horizon_index):
            lCounts_df.loc[-(h+1), self.mCycleResidueName] = None
        df1 = lCounts_df.reset_index()
        df3 = df1.merge(df2 , how='left', on=('index' , 'index'))
        df4 = df3.fillna(method='ffill')
        df4['forecast'] = df4['forecast'] + self.mOffset
        return df4
        
    def fit(self):
        #  print("ESTIMATE_CROSTON_MODEL_START" , self.mCycleResidueName);

        self.set_name();
        
        series = self.mCycleResidueName; 
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mOffset = self.mARFrame[self.mCycleResidueName].min()
        self.mFeatureSelector =  None;
        self.mInputNamesAfterSelection = self.mInputNames;
        self.mComplexity = 2

        lPredicted = self.croston(self.mARFrame);
        self.mARFrame[self.mOutName] = lPredicted['forecast']
        self.mARFrame[self.mOutName + '_residue'] =  self.mARFrame[series] - self.mARFrame[self.mOutName]

        # print("ESTIMATE_CROSTON_MODEL_END" , self.mOutName);


    def transformDataset(self, df, horizon_index = 1):
        series = self.mCycleResidueName;
        pred = self.croston(df, horizon_index)
        df[self.mOutName] = pred['forecast'];
        target = df[series].values
        df[self.mOutName + '_residue'] = target - df[self.mOutName].values        
        return df;

