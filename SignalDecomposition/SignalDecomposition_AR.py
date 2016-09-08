import pandas as pd
import numpy as np
import datetime

from . import SignalDecomposition_Time as tsti
from . import SignalDecomposition_Perf as tsperf
from . import SignalDecomposition_Plots as tsplot

import sklearn as skl
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

class cAbstractAR:
    def __init__(self , cycle_residue_name):
        self.mTimeInfo = tsti.cTimeInfo()
        self.mCycleFrame = pd.DataFrame()
        self.mARFrame = pd.DataFrame()        
        self.mCycleResidueName = cycle_residue_name


    def plot(self):
        tsplot.decomp_plot(self.mARFrame, self.mTimeInfo.mNormalizedTimeColumn,
                           self.mCycleResidueName, self.mOutName , self.mOutName + '_residue');

    def computePerf(self):
        self.mARFitPerf= tsperf.cPerf();
        self.mARForecastPerf= tsperf.cPerf();
        (lFrameFit, lFrameForecast, lFrameTest) = self.mTimeInfo.cutFrame(self.mARFrame);
        self.mARFitPerf.compute(
            lFrameFit[self.mCycleResidueName], lFrameFit[self.mOutName], self.mOutName)
        self.mARForecastPerf.compute(
            lFrameForecast[self.mCycleResidueName], lFrameForecast[self.mOutName], self.mOutName)

class cZeroAR(cAbstractAR):
    def __init__(self , cycle_residue_name):
        super().__init__(cycle_residue_name)
        self.mOutName = self.mCycleResidueName +  '_NoAR'
        self.mFormula = "NoAR";
        
    def fit(self):
        series = self.mCycleResidueName; 
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mTimeInfo.addVars(self.mARFrame);
        self.mARFrame[series] = self.mCycleFrame[series]
        self.mARFrame[self.mOutName] = self.mARFrame[series] * 0.0;
        self.mARFrame[self.mOutName + '_residue'] = self.mARFrame[series];
                

    def transformDataset(self, df):
        series = self.mCycleResidueName; 
        df[self.mOutName] = 0.0;
        target = df[series].values
        df[self.mOutName + '_residue'] = target - df[self.mOutName].values        
        return df;


class cAutoRegressiveModel(cAbstractAR):
    def __init__(self , cycle_residue_name):
        super().__init__(cycle_residue_name)
        self.mARRidge = linear_model.Ridge()

    
    def generateLags(self, df, P):
        self.mARLagNames = [];
        lag_df = pd.DataFrame()
        series = self.mCycleResidueName; 
        lag_df[series] = df[series]
        for i in range(1,P+1):
            name = series+'_Lag' + str(i);
            lag_df[name] = df[series].shift(i)
            self.mARLagNames = self.mARLagNames + [name];
        return lag_df;
    
    def fit(self):
        series = self.mCycleResidueName; 
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mNbLags = int(self.mCycleFrame.shape[0] / 4)
        if(self.mNbLags > self.mOptions.mMaxAROrder):
            self.mNbLags = self.mOptions.mMaxAROrder;
        self.mOutName = self.mCycleResidueName +  '_AR(' + str(self.mNbLags) + ")"; 
        self.mARFrame = self.generateLags(self.mCycleFrame, self.mNbLags);
        self.mAREstimFrame = self.mTimeInfo.getEstimPart(self.mARFrame)
        self.mDefaultValue = self.mAREstimFrame[series].mean()
        self.mARFrame.fillna(self.mDefaultValue , inplace=True)
        self.mAREstimFrame.fillna(self.mDefaultValue , inplace=True)
        
        lARInputs = self.mAREstimFrame.drop([series]  , axis=1).values
        lARTarget = self.mAREstimFrame[series].values
        self.mARRidge.fit(lARInputs, lARTarget)
        
        lARInputsFull = self.mARFrame.drop([series]  , axis=1).values
        self.mARFrame[self.mOutName] = self.mARRidge.predict(lARInputsFull)
        self.mARFrame[self.mOutName + '_residue'] =  self.mARFrame[series] - self.mARFrame[self.mOutName]
        self.mFormula = "AR(" + str(self.mNbLags) + ")";

    def transformDataset(self, df):
        series = self.mCycleResidueName; 
        lag_df = self.generateLags(df, self.mNbLags);
        lag_df.fillna(self.mDefaultValue , inplace=True)
        inputs = lag_df.drop([series]  , axis=1).values
        pred = self.mARRidge.predict(inputs)
        df[self.mOutName] = pred;
        target = df[series].values
        df[self.mOutName + '_residue'] = target - df[self.mOutName].values        
        return df;
        

class cAutoRegressiveEstimator:
    def __init__(self):
        self.mTimeInfo = tsti.cTimeInfo()
        self.mCycleFrame = pd.DataFrame()
        self.mARFrame = pd.DataFrame()
        self.mARList = {}

    def plotAR(self):
        for trend in self.mTrendList:
            for cycle in self.mCycleList[trend]:
                cycle_residue = cycle.getCycleResidueName();
                for autoreg in self.mARList[cycle_residue]:
                    autoreg.plot(); 


    def estimate(self):
        mARList = {}
        for trend in self.mTrendList:
            for cycle in self.mCycleList[trend]:
                cycle_residue = cycle.getCycleResidueName();
                self.mARList[cycle_residue] = [ cZeroAR(cycle_residue)];
                if(self.mOptions.mEnableARModels):
                    if((self.mCycleFrame[cycle_residue].shape[0] > 12) and (self.mCycleFrame[cycle_residue].std() > 0.00001)):
                        self.mARList[cycle_residue] = self.mARList[cycle_residue] + [cAutoRegressiveModel(cycle_residue)];
                for autoreg in self.mARList[cycle_residue]:
                    autoreg.mOptions = self.mOptions;
                    autoreg.mCycleFrame = self.mCycleFrame;
                    autoreg.mTimeInfo = self.mTimeInfo;
                    autoreg.fit();
                    autoreg.computePerf();

