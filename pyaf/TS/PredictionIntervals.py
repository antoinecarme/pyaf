# Copyright (C) 2016 Antoine Carme <Antoine.Carme@outlook.com>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np

from . import SignalDecomposition as sigdec

from . import Perf as tsperf
from . import Utils as tsutil

class cPredictionIntervalsEstimator:
    
    def __init__(self):
        self.mModel = None;
        self.mSignalFrame = None
        self.mHorizon = -1;
        self.mFitPerformances = {}
        self.mForecastPerformances = {}
        self.mTestPerformances = {}
        self.mComputeAllPerfs = True

    def compute_one_perf(self, signal, estimator, iHorizonName):
        # Investigate Large Horizon Models #213 : generate all prediction intervals for all models.
        # Don't compute all the perf indicators for the model selection (AUC is not relevant here, speed issues).
        # Compute all the perf indicators for the selected model at the end of training.
        lPerf = tsperf.cPerf();
        if(self.mComputeAllPerfs):
            lPerf.compute(signal, estimator, iHorizonName);
        else:
            lCriterions = [ self.mModel.mTimeInfo.mOptions.mModelSelection_Criterion ]
            lDict = lPerf.computeCriterionValues(signal, estimator, lCriterions, iHorizonName);
            return lDict
        return lPerf
        
    def computePerformances(self):
        # lTimer = tsutil.cTimer(("cPredictionIntervalsEstimator::computePerformances",
        #                        {"Model" : self.mModel.mOutName, "Horizon" : self.mModel.mTimeInfo.mHorizon}))
        self.mTime = self.mModel.mTime;
        self.mSignal = self.mModel.mOriginalSignal;
        self.mHorizon = self.mModel.mTimeInfo.mHorizon;
        lTimeColumn = self.mTime;
        lSignalColumn = self.mSignal;
        lForecastColumn = str(self.mSignal) + "_Forecast";
        df = self.mModel.mTrend.mSignalFrame.reset_index();
        N = df.shape[0];
        (lOriginalFit, lOriginalForecast, lOriginalTest) = self.mModel.mTimeInfo.mSplit.cutFrame(df);
        df1 = df;
        for h in range(0 , self.mHorizon):
            df2 = None;
            df2 = self.mModel.forecastOneStepAhead(df1, horizon_index = h+1, perf_mode = True);
            df2 = df2.head(N);
            lHorizonName = lForecastColumn + "_" + str(h + 1);
            (lFrameFit, lFrameForecast, lFrameTest) = self.mModel.mTimeInfo.mSplit.cutFrame(df2);
            self.mFitPerformances[lHorizonName] = self.compute_one_perf(lOriginalFit[lSignalColumn], lFrameFit[lForecastColumn], lHorizonName);
            
            self.mForecastPerformances[lHorizonName] = self.compute_one_perf(lOriginalForecast[lSignalColumn], lFrameForecast[lForecastColumn], lHorizonName);
            if(lOriginalTest.shape[0] > 0):
                self.mTestPerformances[lHorizonName] = self.compute_one_perf(lOriginalTest[lSignalColumn], lFrameTest[lForecastColumn], lHorizonName);
            df1 = df2[[lTimeColumn , lForecastColumn,
                       self.mModel.mTimeInfo.mRowNumberColumn,
                       self.mModel.mTimeInfo.mNormalizedTimeColumn]];
            df1.columns = [lTimeColumn , lSignalColumn, self.mModel.mTimeInfo.mRowNumberColumn,
                           self.mModel.mTimeInfo.mNormalizedTimeColumn]
        # self.dump_detailed();

    def dump_detailed(self):
        logger = tsutil.get_pyaf_logger();
        lForecastColumn = str(self.mSignal) + "_Forecast";
        for h in range(0 , self.mHorizon):
            lHorizonName = lForecastColumn + "_" + str(h + 1);
            hn = lHorizonName;
            logger.info("CONFIDENCE_INTERVAL_DUMP_FIT " +str(hn) + " " + str(self.mFitPerformances[hn].mL2) + " " + str(self.mFitPerformances[hn].mMAPE));
            logger.info("CONFIDENCE_INTERVAL_DUMP_FORECAST " +str(hn) + " " + str(self.mForecastPerformances[hn].mL2) + " " + str(self.mForecastPerformances[hn].mMAPE));
            logger.info("CONFIDENCE_INTERVAL_DUMP_TEST " +str(hn) + " " + str(self.mTestPerformances[hn].mL2) + " " + str(self.mTestPerformances[hn].mMAPE));


    def dump(self):
        logger = tsutil.get_pyaf_logger();
        lForecastColumn = str(self.mSignal) + "_Forecast";
        for h in range(0 , self.mHorizon):
            lHorizonName = lForecastColumn + "_" + str(h + 1);
            hn = lHorizonName;
            logger.info("CONFIDENCE_INTERVAL_DUMP_FORECAST " + str(hn) + " " + str(self.mForecastPerformances[hn].mL2));
            
