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
    def __init__(self , cycle_residue_name, P, iExogenous = []):
        super().__init__(cycle_residue_name)
        self.mARRidge = linear_model.Ridge()
        self.mNbLags = P;
        self.mExogenousVariables = iExogenous;
        self.mDefaultValues = {};
        self.mLagOrigins = {};

    def getDefaultValue(self, lag):
        return self.mDefaultValues[lag];

    def addLag(self, df, lag_df, series, p):
        lag_df[series] = df[series]
        name = series+'_Lag' + str(p);
        lag_df[name] = df[series].shift(p)
        self.mARLagNames = self.mARLagNames + [name];        
        self.mLagOrigins[name] = series;
        
    def generateLags(self, df, P):
        self.mARLagNames = [];
        lag_df = pd.DataFrame()
        for p in range(1,P+1):
            # signal lags ... plain old AR model
            self.addLag(df, lag_df, self.mCycleResidueName, p);
            # Exogenous variables lags
            for ex in self.mExogenousVariables:
                self.addLag(df, lag_df, ex, p);
        self.mNonUsedInputs = [self.mCycleResidueName];
        for ex in self.mExogenousVariables:
            self.mNonUsedInputs = self.mNonUsedInputs + [ex]
        return lag_df;
    
    def fit(self):
        series = self.mCycleResidueName; 
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        if(self.mNbLags > self.mOptions.mMaxAROrder):
            self.mNbLags = self.mOptions.mMaxAROrder;
        self.mOutName = self.mCycleResidueName +  '_AR(' + str(self.mNbLags) + ")";
        if(len(self.mExogenousVariables) > 0):
            self.mOutName = self.mCycleResidueName +  '_ARX(' + str(self.mNbLags) + ")";
        self.mARFrame = self.generateLags(self.mCycleFrame, self.mNbLags);
        self.mAREstimFrame = self.mTimeInfo.getEstimPart(self.mARFrame)
        for lag in self.mARLagNames:
            self.mDefaultValues[lag] = self.mAREstimFrame[ self.mLagOrigins[lag] ].mean()
            self.mARFrame[lag].fillna(self.mDefaultValues[lag] , inplace=True)
            self.mAREstimFrame[lag].fillna(self.mDefaultValues[lag] , inplace=True)

        # print("mAREstimFrame columns :" , self.mAREstimFrame.columns);
        lARInputs = self.mAREstimFrame.drop(self.mNonUsedInputs  , axis=1).values
        lARTarget = self.mAREstimFrame[series].values
        self.mARRidge.fit(lARInputs, lARTarget)
        
        lARInputsFull = self.mARFrame.drop(self.mNonUsedInputs  , axis=1).values
        self.mARFrame[self.mOutName] = self.mARRidge.predict(lARInputsFull)
        self.mARFrame[self.mOutName + '_residue'] =  self.mARFrame[series] - self.mARFrame[self.mOutName]
        self.mFormula = "AR(" + str(self.mNbLags) + ")";
        if(len(self.mExogenousVariables) > 0):
            self.mFormula = "ARX(" + str(self.mNbLags) + ")";

    def transformDataset(self, df):
        series = self.mCycleResidueName; 
        lag_df = self.generateLags(df, self.mNbLags);
        for lag in self.mARLagNames:
            lag_df[lag].fillna(self.mDefaultValues[lag] , inplace=True)
        inputs = lag_df.drop(self.mNonUsedInputs , axis=1).values
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
        self.mExogenousVariables = [];
        
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
                if(self.mOptions.mEnableARModels or self.mOptions.mEnableARXModels):
                    if((self.mCycleFrame[cycle_residue].shape[0] > 12) and (self.mCycleFrame[cycle_residue].std() > 0.00001)):
                        N = 10;
                        lStep = self.mCycleFrame.shape[0] / (N*4.0);
                        for n in range(1, 10):
                            lLags = int(lStep * n);
                            if(lLags > 1):
                                if(self.mOptions.mEnableARModels):
                                    lAR = cAutoRegressiveModel(cycle_residue, lLags);
                                    self.mARList[cycle_residue] = self.mARList[cycle_residue] + [lAR];
                                if(self.mOptions.mEnableARXModels):
                                    lARX = cAutoRegressiveModel(cycle_residue, lLags, self.mExogenousVariables);
                                    self.mARList[cycle_residue] = self.mARList[cycle_residue] + [lARX];

                for autoreg in self.mARList[cycle_residue]:
                    autoreg.mOptions = self.mOptions;
                    autoreg.mCycleFrame = self.mCycleFrame;
                    autoreg.mTimeInfo = self.mTimeInfo;
                    autoreg.fit();
                    autoreg.computePerf();

