# Copyright (C) 2016 Antoine Carme <Antoine.Carme@outlook.com>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license


import pandas as pd
import numpy as np

from . import Time as tsti
from . import Perf as tsperf
from . import Plots as tsplot
from . import Utils as tsutil
from . import Complexity as tscomplex


import sklearn.linear_model as linear_model

class cAbstractTrend:
    def __init__(self):
        self.mSignalFrame = None
        self.mTimeInfo = tsti.cTimeInfo()
        self.mTrendFrame = None
        self.mTrendPerf = tsperf.cPerf();
        self.mOutName = None
        self.mFormula = None;
        self.mComplexity = tscomplex.eModelComplexity.High;

    def check_not_nan(self, sig , name):
        #print("check_not_nan");
        if(np.isnan(sig[:-1]).any() or np.isinf(sig[:-1]).any() ):
            logger = tsutil.get_pyaf_logger();
            logger.error("TREND_RESIDUE_WITH_NAN_IN_SIGNAL" + str(sig));
            raise tsutil.Internal_PyAF_Error("INVALID_COLUMN _FOR_TREND_RESIDUE ['"  + name + "'");
        pass


    def computePerf(self):
        if(self.mOptions.mDebug):
            self.check_not_nan(self.mTrendFrame[self.mOutName + '_residue'], self.mOutName + '_residue')
        # self.mTrendFrame.to_csv(self.mOutName + '_residue' + ".csv");

        self.mTrendFitPerf = tsperf.cPerf();
        self.mTrendForecastPerf = tsperf.cPerf();
        (lFrameFit, lFrameForecast, lFrameTest) = self.mSplit.cutFrame(self.mTrendFrame);
        self.mTrendFitPerf.computeCriterionValues(lFrameFit[self.mSignal] ,
                                                  lFrameFit[self.mOutName],
                                                  [self.mTimeInfo.mOptions.mModelSelection_Criterion],
                                                  self.mOutName)
        self.mTrendForecastPerf.computeCriterionValues(lFrameForecast[self.mSignal] ,
                                                       lFrameForecast[self.mOutName],
                                                       [self.mTimeInfo.mOptions.mModelSelection_Criterion],
                                                       self.mOutName)


    def compute_trend_residue(self, df):
        target = df[self.mSignal]
        lTrend = df[self.mOutName]
        if(self.mDecompositionType in ['T+S+R']):
            df[self.mOutName + '_residue'] = target - lTrend
        else:
            # This is questionable. But if only a few values are zero, it is the safest.
            lTrendWithNoZero = lTrend.apply(lambda x : x if(abs(x) > 1e-2) else 1e-2)
            df[self.mOutName + '_residue'] = target / lTrendWithNoZero

    def addTrendInputVariables(self):
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mOutName = self.mSignal + "_" + self.mOutName;
        self.mTrendFrame = pd.DataFrame(index = self.mTimeInfo.mSignalFrame.index)
        self.mTimeInfo.addVars(self.mTrendFrame);

    def fit(self):
        self.fit_specific()
        self.mTrendFrame[self.mOutName] = self.compute(self.mTrendFrame)
        self.compute_trend_residue(self.mTrendFrame)

    def transformDataset(self, df):
        df[self.mOutName] = self.compute(df)
        self.compute_trend_residue(df)
        return df;

    def compute(self, df):
        assert(0)

                
class cConstantTrend(cAbstractTrend):
    def __init__(self):
        cAbstractTrend.__init__(self);
        self.mMean = 0.0
        self.mOutName = "ConstantTrend"
        self.mFormula = self.mOutName;    
        self.mComplexity = tscomplex.eModelComplexity.Low;
        
    def fit_specific(self):
        self.mMean = self.mSplit.getEstimPart(self.mTrendFrame)[self.mSignal].mean()
    
    def compute(self, df):
        Y_pred = self.mMean * np.ones_like(df[self.mSignal]);
        return Y_pred

    def dump_values(self):
        logger = tsutil.get_pyaf_logger();
        logger.info("CONSTANT_TREND " + self.mOutName + " " + str(round(self.mMean, 6)));

class cLag1Trend(cAbstractTrend):
    def __init__(self):
        cAbstractTrend.__init__(self);
        self.mDefaultValue = None
        self.mOutName = "Lag1Trend"
        self.mFormula = self.mOutName;
        self.mComplexity = tscomplex.eModelComplexity.Low;
        
    def replaceFirstMissingValue(self, df, series):
        # print(self.mDefaultValue, type(self.mDefaultValue));
        # Be explicit here .... some integer index does not work.
        df.loc[df.index[0] , series] = self.mDefaultValue;
        # print(df.head());
        
    def fit_specific(self):
        lEstim = self.mSplit.getEstimPart(self.mTrendFrame);
        self.mDefaultValue = lEstim[self.mSignal ].iloc[0]        

    def compute(self, df):
        Y_pred = df[self.mSignal].shift(1)
        Y_pred.loc[df.index[0]] = self.mDefaultValue;
        return Y_pred

    def dump_values(self):
        logger = tsutil.get_pyaf_logger();
        logger.info("LAG1_TREND " + self.mFormula + " " + str(round(self.mDefaultValue, 6)));

class cMovingAverageTrend(cAbstractTrend):
    def __init__(self, iWindow):
        cAbstractTrend.__init__(self);
        self.mOutName = "MovingAverage";
        self.mWindow = iWindow;
        self.mFormula = self.mOutName;
        self.mComplexity = tscomplex.eModelComplexity.Medium;
        
    def fit_specific(self):
        self.mOutName = self.mOutName + "(" + str(self.mWindow) + ")";
        self.mFormula = self.mOutName;
        self.mMean = self.mSplit.getEstimPart(self.mTrendFrame)[self.mSignal].mean()


    def compute(self, df):
        Y_pred = df[self.mSignal].shift(1).rolling(self.mWindow).mean().fillna(method='bfill');
        Y_pred.fillna(self.mMean , inplace=True)
        return Y_pred

    def dump_values(self):
        logger = tsutil.get_pyaf_logger();
        logger.info("MOVING_AVERAGE_TREND " + self.mFormula + " " + str(self.mWindow));

class cMovingMedianTrend(cAbstractTrend):
    def __init__(self, iWindow):
        cAbstractTrend.__init__(self);
        self.mOutName = "MovingMedian";
        self.mWindow = iWindow;
        self.mFormula = self.mOutName;
        self.mComplexity = tscomplex.eModelComplexity.High;
        
    def fit_specific(self):
        self.mOutName = self.mOutName + "(" + str(self.mWindow) + ")";
        self.mFormula = self.mOutName;
        self.mMean = self.mSplit.getEstimPart(self.mTrendFrame)[self.mSignal].mean()


    def compute(self, df):
        Y_pred = df[self.mSignal].shift(1).rolling(self.mWindow).median().fillna(method='bfill');
        Y_pred.fillna(self.mMean , inplace=True)
        return Y_pred

    def dump_values(self):
        logger = tsutil.get_pyaf_logger();
        logger.info("MOVING_MEDIAN_TREND " + self.mFormula + " " + str(self.mWindow));

class cLinearTrend(cAbstractTrend):
    def __init__(self):
        cAbstractTrend.__init__(self);
        self.mTrendRidge = linear_model.Ridge(solver = 'cholesky', alpha = 0.0)
        self.mOutName = "LinearTrend"
        self.mFormula = self.mOutName;
        self.mComplexity = tscomplex.eModelComplexity.Low;

    def fit_specific(self):
        lTrendEstimFrame = self.mSplit.getEstimPart(self.mTrendFrame);
        est_target = lTrendEstimFrame[self.mSignal].values
        est_inputs = lTrendEstimFrame[[self.mTimeInfo.mNormalizedTimeColumn]].values
        self.mTrendRidge.fit(est_inputs, est_target)

    def compute(self, df):
        inputs = df[[self.mTimeInfo.mNormalizedTimeColumn]].values
        Y_pred = self.mTrendRidge.predict(inputs)
        return Y_pred

    def dump_values(self):
        logger = tsutil.get_pyaf_logger();
        logger.info("LINEAR_RIDGE_TREND " + self.mFormula + " " + str((self.mTrendRidge.intercept_.round(6) , self.mTrendRidge.coef_.round(6))));

class cPolyTrend(cAbstractTrend):
    def __init__(self):
        cAbstractTrend.__init__(self);
        self.mTrendRidge = linear_model.Ridge(solver = 'cholesky', alpha = 0.0)
        self.mOutName = "PolyTrend"
        self.mFormula = self.mOutName
        self.mComplexity = tscomplex.eModelComplexity.Medium;

    def fit_specific(self):
        self.mTrendFrame[self.mTimeInfo.mNormalizedTimeColumn + "_^2"] = self.mTrendFrame[self.mTimeInfo.mNormalizedTimeColumn] ** 2;    
        self.mTrendFrame[self.mTimeInfo.mNormalizedTimeColumn + "_^3"] = self.mTrendFrame[self.mTimeInfo.mNormalizedTimeColumn] ** 3;    
        lTrendEstimFrame = self.mSplit.getEstimPart(self.mTrendFrame);
        est_target = lTrendEstimFrame[self.mSignal].values
        est_inputs = lTrendEstimFrame[
            [self.mTimeInfo.mNormalizedTimeColumn,
             self.mTimeInfo.mNormalizedTimeColumn + "_^2",
             self.mTimeInfo.mNormalizedTimeColumn + "_^3"]].values
        self.mTrendRidge.fit(est_inputs, est_target)

    def compute(self, df):
        df[self.mTimeInfo.mNormalizedTimeColumn + "_^2"] = df[self.mTimeInfo.mNormalizedTimeColumn] ** 2;    
        df[self.mTimeInfo.mNormalizedTimeColumn + "_^3"] = df[self.mTimeInfo.mNormalizedTimeColumn] ** 3;    
        inputs = df[
            [self.mTimeInfo.mNormalizedTimeColumn,
             self.mTimeInfo.mNormalizedTimeColumn + "_^2",
             self.mTimeInfo.mNormalizedTimeColumn + "_^3"]].values
        Y_pred = self.mTrendRidge.predict(inputs)
        return Y_pred

    def dump_values(self):
        logger = tsutil.get_pyaf_logger();
        logger.info("POLYNOMIAL_RIDGE_TREND " + self.mFormula + " " + str((self.mTrendRidge.intercept_.round(6) , self.mTrendRidge.coef_.round(6))));

class cTrendEstimator:
    
    def __init__(self):
        self.mSignalFrame = None
        self.mTimeInfo = tsti.cTimeInfo()
        self.mTrendFrame = None


    def needMovingTrend(self, df, i):
        N = df.shape[0];
        if(N < (12 * i)) :
            return False;
        return True;
        
    def defineTrends(self):

        self.mTrendList = [];
        
        if(self.mOptions.mActiveTrends['ConstantTrend']):
            self.mTrendList = [cConstantTrend()];
        
        if(self.mOptions.mActiveTrends['Lag1Trend']):
            self.mTrendList = self.mTrendList + [cLag1Trend()];

        N = self.mSignalFrame.shape[0];
        
        if(N > 1 and self.mOptions.mActiveTrends['LinearTrend']):
            self.mTrendList = self.mTrendList + [cLinearTrend()]

        if(N > 2 and self.mOptions.mActiveTrends['PolyTrend']):
            self.mTrendList = self.mTrendList + [cPolyTrend()]
                
        if(N > 2 and self.mOptions.mActiveTrends['MovingAverage']):
            for i in self.mOptions.mMovingAverageLengths:
                if(self.needMovingTrend(self.mSignalFrame , i)):
                    self.mTrendList = self.mTrendList + [cMovingAverageTrend(i)]

        if(N > 2 and self.mOptions.mActiveTrends['MovingMedian']):
            for i in self.mOptions.mMovingMedianLengths:
                if(self.needMovingTrend(self.mSignalFrame , i)):
                    self.mTrendList = self.mTrendList + [cMovingMedianTrend(i)]
        if(len(self.mTrendList) == 0):
            self.mTrendList = [cConstantTrend()];
            
        # logger = tsutil.get_pyaf_logger();
        # logger.info("ACTIVE_TRENDS" + str(self.mOptions.mActiveTrends));
        # logger.info("TRENDS" + str([tr.mOutName for tr in self.mTrendList]));


        
    def plotTrend(self):
        for trend in self.mTrendList:
            tsplot.decomp_plot(self.mTrendFrame, self.mTimeInfo.mNormalizedTimeColumn, self.mSignal, trend.mOutName , trend.mOutName + '_residue', horizon = self.mTimeInfo.mHorizon);
            

    def addTrendInputVariables(self):
        for trend in self.mTrendList:
            trend.addTrendInputVariables()
        pass

    def check_residue(self , trend, sig, name):
        # print("check_trend_residue ", (name, trend.mDecompositionType, sig.min(), sig.max(), sig.mean(), sig .std()))
        if(np.isnan(sig).any()):
            raise tsutil.Internal_PyAF_Error("Invalid residue_is_nan '" +
                                             str((name, trend.mDecompositionType, sig.min(), sig.max(), sig.mean(), sig .std())) + "'");
        if(sig.max() > 1.e5):
            raise tsutil.Internal_PyAF_Error("Invalid residue_too_large '" +
                                             str((name, trend.mDecompositionType, sig.min(), sig.max(), sig.mean(), sig .std())) + "'");
        pass

    def estimateTrends(self):
        lTimer = None
        if(self.mOptions.mDebugProfile):
            lTimer = tsutil.cTimer(("TRAINING_TRENDS", {"Signal" : self.mTimeInfo.mSignal}))
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mTrendFrame = pd.DataFrame(index = self.mSignalFrame.index)
        self.mTimeInfo.addVars(self.mTrendFrame);
        for trend in self.mTrendList:
            trend.mOptions = self.mOptions
            trend.mDecompositionType = self.mDecompositionType
            trend.mTrendFrame = self.mTrendFrame.copy()
            trend.fit();
            if(trend.mOptions.mDebugPerformance):
                trend.computePerf();
            self.mTrendFrame[trend.mOutName] = trend.mTrendFrame[trend.mOutName]
            self.mTrendFrame[trend.mOutName + "_residue"] = trend.mTrendFrame[trend.mOutName + "_residue"]
            if(self.mOptions.mDebug):
                self.check_residue(trend, self.mTrendFrame[trend.mOutName + "_residue"].values[:-1],
                                   trend.mOutName + "_residue");
        pass

    def estimateTrend(self):
        self.defineTrends();
        for trend in self.mTrendList:
            trend.mSignalFrame = self.mSignalFrame;
            trend.mTimeInfo = self.mTimeInfo;            
            trend.mSplit = self.mSplit
        self.addTrendInputVariables();
        self.estimateTrends()
        for trend in self.mTrendList:
            del trend.mTrendFrame
