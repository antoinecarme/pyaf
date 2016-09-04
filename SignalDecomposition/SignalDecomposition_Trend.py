
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

class cAbstractTrend:
    def __init__(self):
        self.mSignalFrame = pd.DataFrame()
        self.mTimeInfo = tsti.cTimeInfo()
        self.mTrendFrame = pd.DataFrame()
        self.mTrendPerf = tsperf.cPerf();
        self.mOutName = ""
        self.mFormula = None;    

    def computePerf(self):
        self.mTrendFitPerf = tsperf.cPerf();
        self.mTrendForecastPerf = tsperf.cPerf();
        (lFrameFit, lFrameForecast, lFrameTest) = self.mTimeInfo.cutFrame(self.mTrendFrame);
        self.mTrendFitPerf.compute(lFrameFit[self.mSignal] ,
                                   lFrameFit[self.mOutName], self.mOutName)
        self.mTrendForecastPerf.compute(lFrameForecast[self.mSignal] ,
                                        lFrameForecast[self.mOutName], self.mOutName)


class cConstantTrend(cAbstractTrend):
    def __init__(self):
        self.mMean = 0.0
        self.mOutName = "ConstantTrend"
        self.mFormula = self.mOutName;    
        
    def addTrendInputVariables(self):
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mOutName = self.mSignal + "_" + self.mOutName;
        self.mTrendFrame = pd.DataFrame()
        self.mTimeInfo.addVars(self.mTrendFrame);
        self.mTrendEstimFrame = self.mTimeInfo.getEstimPart(self.mTrendFrame);
        self.mMean = self.mTrendEstimFrame[self.mSignal].mean()

    def transformDataset(self, df):
        target = df[self.mSignal].values
        df[self.mOutName] = self.mMean * np.ones_like(df[self.mSignal]);
        df[self.mOutName + '_residue'] = target - df[self.mOutName].values        
        return df;
    
    def fit(self):
        # real lag1
        target = self.mTrendFrame[self.mSignal]
        self.mTrendFrame[self.mOutName] = self.mTrendFrame[self.mSignal].apply(lambda x : self.mMean);
        self.mTrendFrame[self.mOutName + '_residue'] = target - self.mTrendFrame[self.mOutName]
        #self.mTrendFrame.to_csv("aaaa.csv")

    def compute(self):
        Y_pred = self.mMean
        return Y_pred

class cLag1Trend(cAbstractTrend):
    def __init__(self):
        self.mMean = 0.0
        self.mOutName = "Lag1Trend"
        self.mFormula = self.mOutName;
        
    def addTrendInputVariables(self):
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mOutName = self.mSignal + "_" + self.mOutName;
        self.mTrendFrame = pd.DataFrame()
        self.mTimeInfo.addVars(self.mTrendFrame);

    def fit(self):
        # real lag1
        target = self.mTrendFrame[self.mSignal].values
        self.mTrendFrame[self.mOutName] = self.mTrendFrame[self.mSignal].shift(1)
        # all except the horizon
        self.mMean = self.mTimeInfo.getEstimPart(self.mTrendFrame)[self.mSignal].mean()
        self.mTrendFrame[self.mOutName].fillna(self.mMean , inplace=True)
        self.mTrendFrame[self.mOutName + '_residue'] =  target - self.mTrendFrame[self.mOutName].values


    def transformDataset(self, df):
        target = df[self.mSignal].values
        df[self.mOutName] = df[self.mSignal].shift(1).fillna(self.mMean);
        df[self.mOutName + '_residue'] = target - df[self.mOutName].values        
        return df;

    def compute(self):
        Y_pred = self.mTrendFrame[self.mSignal].shift(1)
        return Y_pred


class cMovingAverageTrend(cAbstractTrend):
    def __init__(self, iWindow):
        self.mOutName = "MovingAverage(" + str(iWindow) + ")";
        self.mWindow = iWindow;
        self.mFormula = self.mOutName;
        
    def addTrendInputVariables(self):
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mOutName = self.mSignal + "_" + self.mOutName;
        self.mTrendFrame = pd.DataFrame()
        self.mTimeInfo.addVars(self.mTrendFrame);

    def fit(self):
        # real lag1
        target = self.mTrendFrame[self.mSignal].values
        self.mTrendFrame[self.mOutName] = self.mTrendFrame[self.mSignal].shift(1).rolling(self.mWindow).mean().fillna(method='bfill')
        mean = self.mTimeInfo.getEstimPart(self.mTrendFrame)[self.mSignal].mean()
        self.mTrendFrame[self.mOutName].fillna(mean , inplace=True)
        self.mTrendFrame[self.mOutName + '_residue'] =  target - self.mTrendFrame[self.mOutName].values

    def transformDataset(self, df):
        target = df[self.mSignal].values
        df[self.mOutName] = df[self.mSignal].shift(1).rolling(self.mWindow).mean().fillna(method='bfill');
        df[self.mOutName + '_residue'] = target - df[self.mOutName].values        
        return df;

    def compute(self):
        Y_pred = self.mTrendFrame[self.mSignal].shift(1)
        return Y_pred


class cMovingMedianTrend(cAbstractTrend):
    def __init__(self, iWindow):
        self.mOutName = "MovingMedian(" + str(iWindow) + ")";
        self.mWindow = iWindow;
        self.mFormula = self.mOutName;
        
    def addTrendInputVariables(self):
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mOutName = self.mSignal + "_" + self.mOutName;
        self.mTrendFrame = pd.DataFrame()
        self.mTimeInfo.addVars(self.mTrendFrame);

    def fit(self):
        # real lag1
        target = self.mTrendFrame[self.mSignal].values
        self.mTrendFrame[self.mOutName] = self.mTrendFrame[self.mSignal].shift(1).rolling(self.mWindow).median().fillna(method='bfill')
        mean = self.mTimeInfo.getEstimPart(self.mTrendFrame)[self.mSignal].mean()
        self.mTrendFrame[self.mOutName].fillna(mean , inplace=True)
        self.mTrendFrame[self.mOutName + '_residue'] =  target - self.mTrendFrame[self.mOutName].values

    def transformDataset(self, df):
        target = df[self.mSignal].values
        df[self.mOutName] = df[self.mSignal].shift(1).rolling(self.mWindow).median().fillna(method='bfill');
        df[self.mOutName + '_residue'] = target - df[self.mOutName].values        
        return df;

    def compute(self):
        Y_pred = self.mTrendFrame[self.mSignal].shift(1)
        return Y_pred


class cLinearTrend(cAbstractTrend):
    def __init__(self):
        self.mTrendRidge = linear_model.Ridge()
        self.mOutName = "LinearTrend"
        self.mFormula = self.mOutName;

    def addTrendInputVariables(self):
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mOutName = self.mSignal + "_" + self.mOutName;
        self.mTrendFrame = pd.DataFrame()
        self.mTimeInfo.addVars(self.mTrendFrame);
        self.mTrendEstimFrame = self.mTimeInfo.getEstimPart(self.mTrendFrame);

    def fit(self):
        est_target = self.mTrendEstimFrame[self.mSignal].values
        est_inputs = self.mTrendEstimFrame[[self.mTimeInfo.mNormalizedTimeColumn]].values
        self.mTrendRidge.fit(est_inputs, est_target)
        self.mTrendRidge.score(est_inputs, est_target)
        target = self.mTrendFrame[self.mSignal].values
        inputs = self.mTrendFrame[[self.mTimeInfo.mNormalizedTimeColumn]].values
        self.mTrendFrame[self.mOutName] = self.mTrendRidge.predict(inputs)
        self.mTrendFrame[self.mOutName + '_residue'] =  target - self.mTrendFrame[self.mOutName].values

    def predict_inputs(self, inputs):
        df = pd.DataFrame([inputs])
        pred = self.mTrendRidge.predict(df.values)
        return pred

    def transformDataset(self, df):
        target = df[self.mSignal].values
        inputs = df[[self.mTimeInfo.mNormalizedTimeColumn]].values
        #        df[self.mOutName] = df[self.mTimeInfo.mNormalizedTimeColumn].apply(self.predict_inputs);
        df[self.mOutName] = self.mTrendRidge.predict(inputs)
        df[self.mOutName + '_residue'] = target - df[self.mOutName].values        
        return df;

    def compute(self):
        lTimeAfterSomeSteps = self.mTimeInfo.nextTime(iSteps)
        lTimeAfterSomeStepsNormalized = self.mTimeInfo.normalizeTime(lTimeAfterSomeSteps)
        df = pd.DataFrame([lTimeAfterSomeStepsNormalized , lTimeAfterSomeStepsNormalized ** 2])
        Y_pred = self.mTrendRidge.predict(df.values)
        return Y_pred


class cPolyTrend(cAbstractTrend):
    def __init__(self):
        self.mTrendRidge = linear_model.Ridge()
        self.mOutName = "PolyTrend"
        self.mFormula = self.mOutName

    def addTrendInputVariables(self):
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mOutName = self.mSignal + "_" + self.mOutName;
        self.mTrendFrame = pd.DataFrame()
        self.mTimeInfo.addVars(self.mTrendFrame);
        self.mTrendFrame[self.mTimeInfo.mNormalizedTimeColumn + "_^2"] = self.mTrendFrame[self.mTimeInfo.mNormalizedTimeColumn] ** 2;    
        self.mTrendFrame[self.mTimeInfo.mNormalizedTimeColumn + "_^3"] = self.mTrendFrame[self.mTimeInfo.mNormalizedTimeColumn] ** 3;    
        self.mTrendEstimFrame = self.mTimeInfo.getEstimPart(self.mTrendFrame);

    def fit(self):
        est_target = self.mTrendEstimFrame[self.mSignal].values
        est_inputs = self.mTrendEstimFrame[
            [self.mTimeInfo.mNormalizedTimeColumn,
             self.mTimeInfo.mNormalizedTimeColumn + "_^2",
             self.mTimeInfo.mNormalizedTimeColumn + "_^3"]].values
        self.mTrendRidge.fit(est_inputs, est_target)
        self.mTrendRidge.score(est_inputs, est_target)
        target = self.mTrendFrame[self.mSignal].values
        inputs = self.mTrendFrame[
            [self.mTimeInfo.mNormalizedTimeColumn,
             self.mTimeInfo.mNormalizedTimeColumn + "_^2",
             self.mTimeInfo.mNormalizedTimeColumn + "_^3"]].values
        self.mTrendFrame[self.mOutName] = self.mTrendRidge.predict(inputs)
        self.mTrendFrame[self.mOutName + '_residue'] =  target - self.mTrendFrame[self.mOutName].values


    def transformDataset(self, df):
        df[self.mTimeInfo.mNormalizedTimeColumn + "_^2"] = df[self.mTimeInfo.mNormalizedTimeColumn] ** 2;    
        df[self.mTimeInfo.mNormalizedTimeColumn + "_^3"] = df[self.mTimeInfo.mNormalizedTimeColumn] ** 3;    
        target = df[self.mSignal].values
        inputs = df[
            [self.mTimeInfo.mNormalizedTimeColumn,
             self.mTimeInfo.mNormalizedTimeColumn + "_^2",
             self.mTimeInfo.mNormalizedTimeColumn + "_^3"]].values
        #print(inputs);
        pred = self.mTrendRidge.predict(inputs)
        df[self.mOutName] = pred;
        df[self.mOutName + '_residue'] = target - df[self.mOutName].values        
        return df;


    def compute(self):
        lTimeAfterSomeSteps = self.mTimeInfo.nextTime(iSteps)
        lTimeAfterSomeStepsNormalized = self.mTimeInfo.normalizeTime(lTimeAfterSomeSteps)
        df = pd.DataFrame([lTimeAfterSomeStepsNormalized , lTimeAfterSomeStepsNormalized ** 2])
        Y_pred = self.mTrendRidge.predict(df.values)
        return Y_pred


class cTrendEstimator:
    
    def __init__(self):
        self.mSignalFrame = pd.DataFrame()
        self.mTimeInfo = tsti.cTimeInfo()
        self.mTrendFrame = pd.DataFrame()


    def needMovingTrend(self, df, i):
        N = df.shape[0];
        if(N < (12 * i)) :
            return False;
        return True;
        
    def defineTrends(self):
        
        self.mTrendList = [cConstantTrend()];
        if(not self.mOptions.mEnableTrends):
            return;
        
        self.mTrendList = self.mTrendList + [cLag1Trend()];

        if(self.mOptions.mEnableTimeBasedTrends):
            self.mTrendList = self.mTrendList + [cLinearTrend(), cPolyTrend()]
                
        if(self.mOptions.mEnableMovingAverageTrends):
            for i in self.mOptions.mMovingAverageLengths:
                if(self.needMovingTrend(self.mSignalFrame , i)):
                    self.mTrendList = self.mTrendList + [cMovingAverageTrend(i)]

        if(self.mOptions.mEnableMovingMedianTrends):
            for i in self.mOptions.mMovingMedianLengths:
                if(self.needMovingTrend(self.mSignalFrame , i)):
                    self.mTrendList = self.mTrendList + [cMovingMedianTrend(i)]

        
    def plotTrend(self):
        for trend in self.mTrendList:
            tsplot.decomp_plot(self.mTrendFrame, self.mTimeInfo.mNormalizedTimeColumn, self.mSignal, trend.mOutName , trend.mOutName + '_residue');
            

    def addTrendInputVariables(self):
        for trend in self.mTrendList:
            trend.addTrendInputVariables()
        pass

    def check_residue(self , sig, name):
#        print("check_not_nan "  + name);
#        print(sig);
        if(np.isnan(sig).any()):
            raise ValueError("Invalid residue '" + name + "'");
        pass

    def estimateTrends(self):
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mTrendFrame = pd.DataFrame()
        self.mTimeInfo.addVars(self.mTrendFrame);
        for trend in self.mTrendList:
            trend.fit();
            trend.computePerf();
            self.mTrendFrame[trend.mOutName] = trend.mTrendFrame[trend.mOutName]
            self.mTrendFrame[trend.mOutName + "_residue"] = trend.mTrendFrame[trend.mOutName + "_residue"]
            self.check_residue(self.mTrendFrame[trend.mOutName + "_residue"].values, trend.mOutName + "_residue");
        pass

    def estimateTrend(self):
        self.defineTrends();
        for trend in self.mTrendList:
            trend.mSignalFrame = self.mSignalFrame;
            trend.mTimeInfo = self.mTimeInfo;            
        self.addTrendInputVariables();
        self.estimateTrends()
        
#    def computeTrend(self, iSteps):
#        lTimeAfterSomeSteps = self.mTimeInfo.nextTime(iSteps)
#        lTimeAfterSomeStepsNormalized = self.mTimeInfo.normalizeTime(lTimeAfterSomeSteps)
#        df = pd.DataFrame([lTimeAfterSomeStepsNormalized , lTimeAfterSomeStepsNormalized ** 2])
#        Y_pred = self.mTrendRidge.predict(df.values)
#        return Y_pred

