# Copyright (C) 2016 Antoine Carme <Antoine.Carme@outlook.com>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np

from . import SignalDecomposition as sigdec

from . import Perf as tsperf
from . import Utils as tsutil
from . import TimeSeries_Cutting as tscut


class cPredictionIntervalsEstimator:
    
    def __init__(self):
        self.mModel = None;
        self.mSignalFrame = None
        self.mHorizon = -1;
        self.mPerformances = {"whole" : {}, "detrended" : {}, "deseasonalized" : {}, "transformed" : {}}
        for lType in self.mPerformances.keys():
            self.mPerformances[lType] = {tscut.eDatasetType.Fit : {}, tscut.eDatasetType.Forecast : {},  tscut.eDatasetType.Test : {}}
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

    def compute_detrended_perfs(self, original_cut_dfs, forecast_cut_dfs, h):
        lTransformedSignalColumn = self.mModel.mSignal;
        lTrendColumn = self.mModel.mTrend.mOutName;
        lSeasonalColumn = self.mModel.mCycle.mOutName;
        lARColumn = self.mModel.mAR.mOutName;
        lForecastColumn = str(self.mOriginalSignal) + "_Forecast";
        lHorizonName = lForecastColumn + "_detrended_" + str(h + 1);
        for (lDataset, lDF) in forecast_cut_dfs.items():
            lOriginalDF = original_cut_dfs[lDataset]
            if(lDF.shape[0] > 0):
                lDetrendedForecast = lDF[lSeasonalColumn] + lDF[lARColumn]
                lTarget = lOriginalDF[lTransformedSignalColumn] - lDF[lTrendColumn]
                if(self.mModel.mDecompositionType == 'TSR'):
                    lDetrendedForecast = lDF[lSeasonalColumn] * lDF[lARColumn]
                lTarget = lOriginalDF[lTransformedSignalColumn] / lDF[lTrendColumn]
                if(self.mModel.mDecompositionType == 'TS+R'):
                    lDetrendedForecast = lDF[lARColumn]
                    lTarget = lOriginalDF[lTransformedSignalColumn] / lDF[lTrendColumn]
                self.mPerformances["detrended"][lDataset][lHorizonName] = self.compute_one_perf(lTarget, lDetrendedForecast, lHorizonName)


    def compute_transformed_perfs(self, original_cut_dfs, forecast_cut_dfs, h):
        lTransformedSignalColumn = self.mModel.mSignal;
        lTrendColumn = self.mModel.mTrend.mOutName;
        lSeasonalColumn = self.mModel.mCycle.mOutName;
        lARColumn = self.mModel.mAR.mOutName;
        lForecastColumn = str(self.mOriginalSignal) + "_Forecast";
        lHorizonName = lForecastColumn + "_transformed_" + str(h + 1);

        for (lDataset, lDF) in forecast_cut_dfs.items():
            lOriginalDF = original_cut_dfs[lDataset]
            if(lDF.shape[0] > 0):
                self.mPerformances["transformed"][lDataset][lHorizonName] = self.compute_one_perf(lOriginalDF[lTransformedSignalColumn] , lDF[lTrendColumn] + lDF[lSeasonalColumn] + lDF[lARColumn], lHorizonName);


    def compute_deseasonalized_perfs(self, original_cut_dfs, forecast_cut_dfs, h):
        lTransformedSignalColumn = self.mModel.mSignal;
        lTrendColumn = self.mModel.mTrend.mOutName;
        lSeasonalColumn = self.mModel.mCycle.mOutName;
        lARColumn = self.mModel.mAR.mOutName;
        lForecastColumn = str(self.mOriginalSignal) + "_Forecast";
        lHorizonName = lForecastColumn + "_deseasonalized_" + str(h + 1);
        
        for (lDataset, lDF) in forecast_cut_dfs.items():
            lOriginalDF = original_cut_dfs[lDataset]
            if(lDF.shape[0] > 0):
                lDeseasonalizedForecast = lDF[lTrendColumn] + lDF[lARColumn]
                lTarget = lOriginalDF[lTransformedSignalColumn] - lDF[lSeasonalColumn]
                if(self.mModel.mDecompositionType == 'TSR'):
                    lDeseasonalizedForecast = lDF[lTrendColumn] * lDF[lARColumn]
                    lTarget = lOriginalDF[lTransformedSignalColumn] / lDF[lSeasonalColumn]
                if(self.mModel.mDecompositionType == 'TS+R'):
                    lDeseasonalizedForecast = lDF[lARColumn]
                    lTarget = lOriginalDF[lTransformedSignalColumn] / lDF[lSeasonalColumn]
                self.mPerformances["deseasonalized"][lDataset][lHorizonName] = self.compute_one_perf(lTarget, lDeseasonalizedForecast, lHorizonName);

    def compute_whole_perfs(self, original_cut_dfs, forecast_cut_dfs, h):
        lOriginalSignalColumn = self.mOriginalSignal;
        lForecastColumn = str(self.mOriginalSignal) + "_Forecast";
        lHorizonName = lForecastColumn + "_" + str(h + 1);
        
        for (lDataset, lDF) in forecast_cut_dfs.items():
            lOriginalDF = original_cut_dfs[lDataset]
            if(lDF.shape[0] > 0):
                self.mPerformances["whole"][lDataset][lHorizonName] = self.compute_one_perf(lOriginalDF[lOriginalSignalColumn], lDF[lForecastColumn], lHorizonName);

        
    def computePerformances(self):
        # lTimer = tsutil.cTimer(("cPredictionIntervalsEstimator::computePerformances",
        #                        {"Model" : self.mModel.mOutName, "Horizon" : self.mModel.mTimeInfo.mHorizon}))
        self.mTime = self.mModel.mTime;
        self.mOriginalSignal = self.mModel.mOriginalSignal;
        self.mHorizon = self.mModel.mTimeInfo.mHorizon;
        lTimeColumn = self.mTime;
        lOriginalSignalColumn = self.mOriginalSignal;
        lForecastColumn = str(self.mOriginalSignal) + "_Forecast";
        df = self.mModel.mTrend.mSignalFrame.reset_index();
        N = df.shape[0];
        original_cut_dfs = self.mModel.mTimeInfo.mSplit.cutFrame(df);
        df1 = df[ [lTimeColumn , lOriginalSignalColumn] ];
        for h in range(0 , self.mHorizon):
            df2 = self.mModel.forecastOneStepAhead(df1, horizon_index = h+1, perf_mode = True);
            df2 = df2.head(N);
            forecast_cut_dfs = self.mModel.mTimeInfo.mSplit.cutFrame(df2);
            self.compute_whole_perfs(original_cut_dfs, forecast_cut_dfs, h)
            # self.compute_detrended_perfs(original_cut_dfs, forecast_cut_dfs, h)
            #self.compute_deseasonalized_perfs(original_cut_dfs, forecast_cut_dfs, h)
            # self.compute_transformed_perfs(original_cut_dfs, forecast_cut_dfs, h)
            
            df1 = df2[[lTimeColumn , lForecastColumn,
                       self.mModel.mTimeInfo.mRowNumberColumn,
                       self.mModel.mTimeInfo.mNormalizedTimeColumn]];
            df1.columns = [lTimeColumn , lOriginalSignalColumn, self.mModel.mTimeInfo.mRowNumberColumn,
                           self.mModel.mTimeInfo.mNormalizedTimeColumn]
        # self.dump_detailed();

    def dump_detailed(self):
        logger = tsutil.get_pyaf_logger();
        lForecastColumn = str(self.mOriginalSignal) + "_Forecast";
        for h in range(0 , self.mHorizon):
            lHorizonName = lForecastColumn + "_" + str(h + 1);
            hn = lHorizonName;
            logger.info("CONFIDENCE_INTERVAL_DUMP_FIT " +str(hn) + " " + str(self.mFitPerformances[hn].mL2) + " " + str(self.mFitPerformances[hn].mMAPE));
            logger.info("CONFIDENCE_INTERVAL_DUMP_FORECAST " +str(hn) + " " + str(self.mForecastPerformances[hn].mL2) + " " + str(self.mForecastPerformances[hn].mMAPE));
            logger.info("CONFIDENCE_INTERVAL_DUMP_TEST " +str(hn) + " " + str(self.mTestPerformances[hn].mL2) + " " + str(self.mTestPerformances[hn].mMAPE));


    def dump(self):
        logger = tsutil.get_pyaf_logger();
        lForecastColumn = str(self.mOriginalSignal) + "_Forecast";
        for h in range(0 , self.mHorizon):
            lHorizonName = lForecastColumn + "_" + str(h + 1);
            hn = lHorizonName;
            logger.info("CONFIDENCE_INTERVAL_DUMP_FORECAST " + str(hn) + " " + str(self.mForecastPerformances[hn].mL2));
            
