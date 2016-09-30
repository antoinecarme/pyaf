import pandas as pd
import numpy as np
import datetime

from . import Time as tsti
from . import Perf as tsperf
from . import Plots as tsplot

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
        self.mComplexity = 4;
        self.mFormula = None;


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
        self.mComplexity = 0;
        
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
    def __init__(self , cycle_residue_name, P, iExogenousInfo = None ):
        super().__init__(cycle_residue_name)
        self.mARRidge = linear_model.Ridge()
        self.mNbLags = P;
        self.mExogenousInfo = iExogenousInfo;
        self.mDefaultValues = {};
        self.mLagOrigins = {};

    def getDefaultValue(self, lag):
        return self.mDefaultValues[lag];

    def addLag(self, df, lag_df, series, p):
        name = series+'_Lag' + str(p);
        lSeries = df[series];
        lShiftedSeries = lSeries.shift(p)
        lDefaultValue = lSeries.iloc[0];
        if(series in self.mDefaultValues.keys()):
            lDefaultValue = self.mDefaultValues[series];
        else:
            self.mDefaultValues[series] = lDefaultValue;
            
        for i in range(p):
            lShiftedSeries.iloc[ i ] = lDefaultValue;
            
        lag_df[name] = lShiftedSeries;
        self.mARLagNames = self.mARLagNames + [name];        
        self.mLagOrigins[name] = series;
        
    def generateLags(self, df, P):
        self.mARLagNames = [];
        lag_df = pd.DataFrame()
        lag_df[self.mCycleResidueName] = df[self.mCycleResidueName]
        for p in range(1,P+1):
            # signal lags ... plain old AR model
            self.addLag(df, lag_df, self.mCycleResidueName, p);
            # Exogenous variables lags
            if(self.mExogenousInfo is not None):
                # print(self.mExogenousInfo.mExogenousDummies);
                # print(df.columns);
                for ex in self.mExogenousInfo.mExogenousDummies:
                    self.addLag(df, lag_df, ex, p);
        return lag_df;
    
    def fit(self):
        series = self.mCycleResidueName; 
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        if(self.mNbLags > self.mOptions.mMaxAROrder):
            self.mNbLags = self.mOptions.mMaxAROrder;
        self.mOutName = self.mCycleResidueName +  '_AR(' + str(self.mNbLags) + ")";
        self.mFormula = "AR(" + str(self.mNbLags) + ")";
        if(self.mExogenousInfo is not None):
            self.mCycleFrame = self.mExogenousInfo.transformDataset(self.mCycleFrame);
            self.mOutName = self.mCycleResidueName +  '_ARX(' + str(self.mNbLags) + ")";
            self.mFormula = "ARX(" + str(self.mNbLags) + ")";
        self.mARFrame = self.generateLags(self.mCycleFrame, self.mNbLags);
        self.mAREstimFrame = self.mTimeInfo.getEstimPart(self.mARFrame)

        # print("mAREstimFrame columns :" , self.mAREstimFrame.columns);
        lARInputs = self.mAREstimFrame.drop([self.mCycleResidueName]  , axis=1).values
        lARTarget = self.mAREstimFrame[series].values
        self.mARRidge.fit(lARInputs, lARTarget)
        
        lARInputsFull = self.mARFrame.drop([self.mCycleResidueName]  , axis=1).values
        self.mARFrame[self.mOutName] = self.mARRidge.predict(lARInputsFull)
        self.mARFrame[self.mOutName + '_residue'] =  self.mARFrame[series] - self.mARFrame[self.mOutName]

    def transformDataset(self, df):
        series = self.mCycleResidueName; 
        if(self.mExogenousInfo is not None):
            df = self.mExogenousInfo.transformDataset(df);
        lag_df = self.generateLags(df, self.mNbLags);
        # lag_df.to_csv("LAGGED_ " + str(self.mNbLags) + ".csv");
        inputs = lag_df.drop([self.mCycleResidueName]  , axis=1).values
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
        self.mExogenousInfo = None;
        
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
                                if(self.mOptions.mEnableARXModels and
                                   (self.mExogenousInfo is not None)):
                                    lARX = cAutoRegressiveModel(cycle_residue, lLags,
                                                                self.mExogenousInfo);
                                    self.mARList[cycle_residue] = self.mARList[cycle_residue] + [lARX];

                for autoreg in self.mARList[cycle_residue]:
                    autoreg.mOptions = self.mOptions;
                    autoreg.mCycleFrame = self.mCycleFrame;
                    autoreg.mTimeInfo = self.mTimeInfo;
                    autoreg.fit();
                    autoreg.computePerf();

