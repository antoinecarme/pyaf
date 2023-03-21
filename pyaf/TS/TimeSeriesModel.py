# Copyright (C) 2016 Antoine Carme <Antoine.Carme@outlook.com>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np

from . import PredictionIntervals as predint

from . import Plots as tsplot
from . import Perf as tsperf
from . import Utils as tsutil
from . import Complexity as tscomplex


def to_str(x):
    return str(np.round(x, 6))

class cTimeSeriesModel:
    
    def __init__(self, transf, iDecompType, trend, cycle, autoreg):
        self.mTransformation = transf;
        self.mDecompositionType = iDecompType;
        self.mTrend = trend;
        self.mCycle = cycle;
        self.mAR = autoreg;
        self.mFitPerformances = {}
        self.mForecastPerformances = {}
        self.mTestPerformances = {}
        self.mOutName = self.mAR.mOutName;
        self.mOriginalSignal = self.mTransformation.mOriginalSignal;
        self.mTimeInfo = self.mTrend.mTimeInfo;
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mTrainingVersionInfo = self.getVersions();

        

    def get_model_category(self):
        lModelCategory = (self.mTransformation.__class__.__name__,
                          self.mTrend.__class__.__name__,
                          self.mCycle.__class__.__name__,
                          self.mAR.__class__.__name__)
        lModelCategory = self.mTransformation.mFormula + "_" + self.mTrend.mFormula + "_" + self.mCycle.mFormula + "_" + self.mAR.mFormula
        return str(lModelCategory)
        
    def getComplexity(self):
        # This is just a way to give priority to additive decompositions (default = 0 for additive).
        lModelTypeComplexity = {
            "T+S+R" : tscomplex.eModelComplexity.Low,
            "TS+R" : tscomplex.eModelComplexity.High,
            "TSR" : tscomplex.eModelComplexity.High,
        }
        lComplexity = {'Decomposition' : lModelTypeComplexity.get(self.mDecompositionType).value,
                       'Transformation' : self.mTransformation.mComplexity.value,
                       'Trend' : self.mTrend.mComplexity.value,
                       'Cycle' : self.mCycle.mComplexity.value,
                       'AR' : self.mAR.mComplexity.value}
        return lComplexity;     

    def getComplexity_as_ordering_string(self):
        lComplexity = self.getComplexity()
        lValues = [str(v) for (k,v) in lComplexity.items()]
        lStr = "".join(sorted(lValues))
        return lStr;     

    def updateAllPerfs(self):
        lTimer = tsutil.cTimer(("UPDATE_BEST_MODEL_PERFS", {"Signal" : self.mOriginalSignal, "Model" : self.mOutName}))
        self.updatePerfs(compute_all_indicators = True)

    def updatePerfs(self, compute_all_indicators = False):
        # Investigate Large Horizon Models #213 : generate all prediction intervals for all models.
        # Keep the model that holds the best perf at the horizon H.
        # Consider perfs a horizon H instead of looking at horizon 1 (WIP ...).
        # Don't compute all the perf indicators for the model selection (AUC is not relevant here, speed issues).
        # Compute all the perf indicators for the selected model at the end of training.

        # lTimer = tsutil.cTimer(("UPDATE_MODEL_PERFS", {"Signal" : self.mOriginalSignal, "Model" : self.mOutName}))
        lPredictionIntervalsEstimator = predint.cPredictionIntervalsEstimator();
        lPredictionIntervalsEstimator.mModel = self;
        lPredictionIntervalsEstimator.mComputeAllPerfs = compute_all_indicators
        lPredictionIntervalsEstimator.computePerformances();
        lForecastColumn = str(self.mOriginalSignal) + "_Forecast";
                    
        self.mFitPerfs = lPredictionIntervalsEstimator.mFitPerformances
        self.mForecastPerfs = lPredictionIntervalsEstimator.mForecastPerformances
        self.mTestPerfs = lPredictionIntervalsEstimator.mTestPerformances

    def get_perfs_summary(self):
        output = {"Fit" : {}, "Forecast" : {}, "Test" : {}}
        lForecastColumn = str(self.mOriginalSignal) + "_Forecast";
        lCriterion = self.mTimeInfo.mOptions.mModelSelection_Criterion
        for h in [1, self.mTimeInfo.mHorizon]:
            lHorizonName = lForecastColumn + "_" + str(h);
            output["Fit"][h] = self.mFitPerfs[lHorizonName]
            output["Forecast"][h] = self.mForecastPerfs[lHorizonName]
            output["Test"][h] = self.mTestPerfs[lHorizonName]
        return output
        
    def aggregate_criteria(self, criteria):
        lAggregated = criteria[0]
        return lAggregated
        
    def get_aggregated_criterion_values_for_model_selection(self):
        lCriterion = self.mTimeInfo.mOptions.mModelSelection_Criterion
        lForecastColumn = str(self.mOriginalSignal) + "_Forecast";
        (lFitCritData, lForecastCritData, lTestCritData) = ([], [], [])
        for h in range(self.mTimeInfo.mHorizon):
            lHorizonName = lForecastColumn + "_" + str(h + 1);
            lFitCritData.append(self.mFitPerfs[lHorizonName].get(lCriterion))
            lForecastCritData.append(self.mForecastPerfs[lHorizonName].get(lCriterion))
            lTestCritData.append(self.mTestPerfs[lHorizonName].get(lCriterion))
        lAggFitCrit = self.aggregate_criteria(lFitCritData)
        lAggForecastCrit = self.aggregate_criteria(lForecastCritData)
        lAggTestCrit = self.aggregate_criteria(lTestCritData)
            
        return (lAggFitCrit, lAggForecastCrit, lAggTestCrit)
        
    def computePredictionIntervals(self):
        # prediction intervals
        if(self.mTimeInfo.mOptions.mAddPredictionIntervals):
            lTimer = tsutil.cTimer(("COMPUTE_PREDICTION_INTERVALS", {"Signal" : self.mOriginalSignal}))
            self.mPredictionIntervalsEstimator = predint.cPredictionIntervalsEstimator();
            self.mPredictionIntervalsEstimator.mModel = self;
            self.mPredictionIntervalsEstimator.mComputeAllPerfs = True;            
            self.mPredictionIntervalsEstimator.computePerformances();

    def getFormula(self):
        if(self.mDecompositionType in ['TS+R']):
            return self.mTrend.mFormula + " * " + self.mCycle.mFormula + " + " + self.mAR.mFormula
        if(self.mDecompositionType in ['TSR']):
            return self.mTrend.mFormula + " * " + self.mCycle.mFormula + " * " + self.mAR.mFormula
        return self.mTrend.mFormula + " + " + self.mCycle.mFormula + " + " + self.mAR.mFormula


    def signal_info(self):
        logger = tsutil.get_pyaf_logger();
        lSignal = self.mTrend.mSignalFrame[self.mOriginalSignal];
        lStr1 = "SignalVariable='" + self.mOriginalSignal +"'";
        lStr1 += " Length=" + str(lSignal.shape[0]) + " ";
        lStr1 += " Min=" + to_str(np.min(lSignal)) + " Max="  + to_str(np.max(lSignal)) + " ";
        lStr1 += " Mean=" + to_str(np.mean(lSignal)) + " StdDev="  + to_str(np.std(lSignal));
        lSignal = self.mTrend.mSignalFrame[self.mSignal];
        lStr2 = "TransformedSignalVariable='" + self.mSignal +"'";
        lStr2 += " Min=" + to_str(np.min(lSignal)) + " Max="  + to_str(np.max(lSignal)) + " ";
        lStr2 += " Mean=" + to_str(np.mean(lSignal)) + " StdDev="  + to_str(np.std(lSignal));
        logger.info("SIGNAL_DETAIL_ORIG " + lStr1);
        logger.info("SIGNAL_DETAIL_TRANSFORMED " + lStr2);
        if(self.mAR.mExogenousInfo):
            logger.info("EXOGENOUS_DATA " + str(self.mAR.mExogenousInfo.mExogenousVariables));        
        return (lStr1 , lStr2);

    def perf_info(self):
        logger = tsutil.get_pyaf_logger();
        lForecastColumn = str(self.mOriginalSignal) + "_Forecast";
        lCriterion = self.mTimeInfo.mOptions.mModelSelection_Criterion
        for h in [1, self.mTimeInfo.mHorizon]:
            lHorizonName = lForecastColumn + "_" + str(h);
            lDict_Fit_H = self.mFitPerfs[lHorizonName].to_dict_summary(lCriterion)
            logger.info("MODEL_PERFS Fit STEP=" + str(h) + " " + str(lDict_Fit_H));
            lDict_Forecast_H = self.mForecastPerfs[lHorizonName].to_dict_summary(lCriterion)
            logger.info("MODEL_PERFS Forecast STEP=" + str(h) + " " + str(lDict_Forecast_H));
            lDict_Test_H = self.mTestPerfs[lHorizonName].to_dict_summary(lCriterion)
            logger.info("MODEL_PERFS Test STEP=" + str(h) + " " + str(lDict_Test_H));

    def decomposition_info(self):
        logger = tsutil.get_pyaf_logger();
        logger.info("DECOMPOSITION_TYPE '" + self.mDecompositionType + "'");
        logger.info("BEST_TRANSOFORMATION_TYPE '" + self.mTransformation.get_name("") + "'");
        logger.info("BEST_DECOMPOSITION  '" + self.mOutName + "' [" + self.getFormula() + "]");
        logger.info("TREND_DETAIL '" + self.mTrend.mOutName + "' [" + self.mTrend.mFormula + "]");
        logger.info("CYCLE_DETAIL '"+ self.mCycle.mOutName + "' [" + self.mCycle.mFormula + "]");
        logger.info("AUTOREG_DETAIL '" + self.mAR.mOutName + "' [" + self.mAR.mFormula + "]");

    def decomposition_detail_info(self):
        logger = tsutil.get_pyaf_logger();
        lComplexityStr = self.getComplexity_as_ordering_string()
        logger.info("MODEL_COMPLEXITY " + str(self.getComplexity()) + " [" + lComplexityStr + "]");
        logger.info("SIGNAL_TRANSFORMATION_DETAIL_START");
        self.mTransformation.dump_values();
        logger.info("SIGNAL_TRANSFORMATION_DETAIL_END");
        logger.info("TREND_DETAIL_START");
        self.mTrend.dump_values();
        logger.info("TREND_DETAIL_END");
        logger.info("CYCLE_MODEL_DETAIL_START");
        self.mCycle.dump_values();
        logger.info("CYCLE_MODEL_DETAIL_END");
        logger.info("AR_MODEL_DETAIL_START");
        self.mAR.dumpCoefficients();
        logger.info("AR_MODEL_DETAIL_END");
        
    
    def getInfo(self):
        logger = tsutil.get_pyaf_logger();
        logger.info("TIME_DETAIL " + self.mTrend.mTimeInfo.info());
        self.signal_info()
        self.decomposition_info()
        self.perf_info()
        self.decomposition_detail_info()

    def compute_model_forecast(self, iTrendValue, iCycleValue, iARValue):
        if(self.mDecompositionType in ['TS+R']):
            return iTrendValue * iCycleValue + iARValue
        if(self.mDecompositionType in ['TSR']):
            lTrendValue = iTrendValue.clip(-100, 100)
            lCycleValue = iCycleValue.clip(-100, 100)
            lARValue = iARValue.clip(-100, 100)
            return lTrendValue * lCycleValue * lARValue
        return iTrendValue + iCycleValue + iARValue

    def forecastOneStepAhead(self , df , horizon_index = 1, perf_mode = False):
        assert(self.mTime in df.columns)
        assert(self.mOriginalSignal in df.columns)
        lPrefix = self.mSignal + "_";
        df1 = df;
        # df1.to_csv("before.csv");
        # add new line with new time value, row_number and nromalized time
        # add signal tranformed column
        df1 = self.mTransformation.transformDataset(df1);
        # df1.to_csv("after_transformation.csv");
        #print("Transformation update : " , df1.columns);

        df1 = self.mTimeInfo.transformDataset(df1);
        # df1.to_csv("after_time.csv");
        # print("TimeInfo update : " , df1.columns);
        # compute the trend based on the transformed column and compute trend residue
        df1 = self.mTrend.transformDataset(df1);
        #print("Trend update : " , df1.columns);
        # df1.to_csv("after_trend.csv");
        # compute the cycle and its residue based on the trend residue
        df1 = self.mCycle.transformDataset(df1);
        # df1.to_csv("after_cycle.csv");
        #print("Cycle update : " , df1.columns);
        # compute the AR componnet and its residue based on the cycle residue
        df1 = self.mAR.transformDataset(df1, horizon_index);
        # df1.to_csv("after_ar.csv");
        #print("AR update : " , df1.columns);
        # compute the forecast and its residue (forecast = trend  + cycle + AR)
        df2 = df1;
        lTrendColumn = df2[self.mTrend.mOutName]
        lCycleColumn = df2[self.mCycle.mOutName]
        lARColumn = df2[self.mAR.mOutName]
        lSignal = df2[self.mSignal]
        if(not perf_mode):
            df2[self.mOriginalSignal + '_Transformed'] =  lSignal;
            df2[lPrefix + 'Trend'] =  lTrendColumn;
            df2[lPrefix + 'Trend_residue'] = df2[self.mCycle.mTrend_residue_name]
            df2[lPrefix + 'Cycle'] =  lCycleColumn;
            df2[lPrefix + 'Cycle_residue'] = df2[self.mCycle.getCycleResidueName()];
            df2[lPrefix + 'AR'] =  lARColumn ;
            df2[lPrefix + 'AR_residue'] = df2[self.mAR.mOutName + '_residue'];

        lPrefix2 = str(self.mOriginalSignal) + "_";
        # print("TimeSeriesModel_forecast_invert");
        df2[lPrefix + 'TransformedForecast'] = self.compute_model_forecast(lTrendColumn, lCycleColumn, lARColumn)
        df2[lPrefix2 + 'Forecast'] = self.mTransformation.invert(df2[lPrefix + 'TransformedForecast']);

        if(not perf_mode):
            df2[lPrefix + 'TransformedResidue'] =  lSignal - df2[lPrefix + 'TransformedForecast']
            lOriginalSignal = df2[self.mOriginalSignal]
            df2[lPrefix2 + 'Residue'] =  lOriginalSignal - df2[lPrefix2 + 'Forecast']
        df2.reset_index(drop=True, inplace=True);
        return df2;


    def forecast_all_horizons(self , df , iHorizon):
        N0 = df.shape[0];
        df1 = self.forecastOneStepAhead(df, 1)
        lForecastColumnName = str(self.mOriginalSignal) + "_Forecast";
        for h in range(0 , iHorizon - 1):
            # print(df1.info());
            N = df1.shape[0];
            # replace the signal with the forecast in the last line  of the dataset
            lPos = df1.index[N - 1];
            lSignal = df1.loc[lPos , lForecastColumnName];
            df1.loc[lPos , self.mOriginalSignal] = lSignal;
            df1 = df1[[self.mTime , self.mOriginalSignal, self.mTimeInfo.mRowNumberColumn, self.mTimeInfo.mNormalizedTimeColumn]];
            df1 = self.forecastOneStepAhead(df1 , h+2)

        assert((N0 + iHorizon) == df1.shape[0])
        N1 = df1.shape[0];
        lPrefix = self.mSignal + "_";
        lFieldsToErase = [ self.mOriginalSignal, self.mSignal, self.mOriginalSignal + "_Transformed", 
                           self.mTrend.mOutName + '_residue', lPrefix + 'Trend_residue',
                           self.mCycle.mOutName + '_residue', lPrefix + 'Cycle_residue',
                           self.mAR.mOutName + '_residue',  lPrefix + 'AR_residue',
                           lPrefix + 'TransformedResidue', str(self.mOriginalSignal) + '_Residue']
        df1.loc[N1 -iHorizon : N1, lFieldsToErase] = np.nan
        return df1

    
    def forecast(self , df , iHorizon):
        df1 = self.forecast_all_horizons(df, iHorizon)
        # print(df.head())
        # print(df1.head())
        if(self.mTimeInfo.mOptions.mAddPredictionIntervals):
            df1 = self.addPredictionIntervals(df, df1, iHorizon);
            self.addForecastQuantiles(df, df1, iHorizon);
        if(self.mTimeInfo.mOptions.mForecastRectifier is not None):
            df1 = self.applyForecastRectifier(df1)
        return df1

    def applyForecastRectifier(self, df):
        df1 = df;
        if(self.mTimeInfo.mOptions.mForecastRectifier == "relu"):
            lForecastColumnName = str(self.mOriginalSignal) + "_Forecast";
            df1[lForecastColumnName] = df1[lForecastColumnName].apply(lambda x : max(x, 0))
        return df1

    def addPredictionIntervals(self, iInputDS, iForecastFrame, iHorizon):
        lSignalColumn = self.mOriginalSignal;

        N = iInputDS.shape[0];
        lForecastColumn = str(lSignalColumn) + "_Forecast";
        lLowerBoundName = lForecastColumn + '_Lower_Bound'
        lUpperBoundName = lForecastColumn + '_Upper_Bound';
        iForecastFrame[lLowerBoundName] = np.nan; 
        iForecastFrame[lUpperBoundName] = np.nan; 

        lConfidence = 1.96 ; # 0.95
        # the prediction intervals are only computed for the training horizon
        lHorizon = min(iHorizon , self.mTimeInfo.mHorizon);
        lWidths = [lConfidence * self.mPredictionIntervalsEstimator.mForecastPerformances[lForecastColumn + "_" + str(h + 1)].mL2
                   for h in range(0 , self.mTimeInfo.mHorizon)]
        lWidths = (lWidths + [np.nan]*iHorizon)[:iHorizon]
        lForcastValues = iForecastFrame.loc[N:N+iHorizon, lForecastColumn]
        # print(lForcastValues.head(lHorizon))
        # print(iHorizon, self.mTimeInfo.mHorizon, lHorizon, lForcastValues.shape)
        # print(lWidths)
        iForecastFrame.loc[N:N+iHorizon, lLowerBoundName] = lForcastValues - lWidths
        iForecastFrame.loc[N:N+iHorizon, lUpperBoundName] = lForcastValues + lWidths
        return iForecastFrame;

    def addForecastQuantiles(self, iInputDS, iForecastFrame, iHorizon):
        lSignalColumn = self.mOriginalSignal;

        N = iInputDS.shape[0] ;
        lForecastColumn = str(lSignalColumn) + "_Forecast";
        lQuantileName = lForecastColumn + "_Quantile_"

        # the prediction intervals are only computed for the training horizon
        lHorizon = min(iHorizon , self.mTimeInfo.mHorizon);
        lPerfs = [self.mPredictionIntervalsEstimator.mForecastPerformances[lForecastColumn + "_" + str(h + 1)]
                   for h in range(0 , self.mTimeInfo.mHorizon)]
        lForcastValues = iForecastFrame.loc[N:N+iHorizon, lForecastColumn]
        
        lQuantiles = self.mPredictionIntervalsEstimator.mForecastPerformances[lForecastColumn + "_1"].mErrorQuantiles.keys()
        lQuantiles = sorted(lQuantiles)
        for q in lQuantiles:
            iForecastFrame[lQuantileName + str(q)] = np.nan
            lQuants = [lPerf.mErrorQuantiles[q] for lPerf in lPerfs]
            lQuants = (lQuants + [np.nan]*iHorizon)[:iHorizon]
            iForecastFrame.loc[N:N+iHorizon, lQuantileName + str(q)] =  lForcastValues.values + np.array(lQuants)
        return iForecastFrame;


    def plotForecasts(self, df):
        lPrefix = self.mSignal + "_";
        lTime = self.mTimeInfo.mNormalizedTimeColumn;            
        tsplot.decomp_plot(df,
                           lTime, lPrefix + 'Signal',
                           lPrefix + 'Forecast' , lPrefix + 'Residue', horizon = self.mTimeInfo.mHorizon, title = self.mOriginalSignal + " Forecasts");

    def to_dict(self, iWithOptions = False):
        dict1 = {};
        d1 = { "Time" : self.mTimeInfo.to_dict(),
               "Signal" : self.mOriginalSignal,
               "Training_Signal_Length" : self.mTimeInfo.mSignalFrame.shape[0]};
        dict1["Dataset"] = d1;
        lTransformation = self.mTransformation.mFormula;
        d2 = { "Best_Decomposition" : self.mOutName,
               "Signal_Decomposition_Type" : self.mDecompositionType,
               "Signal_Transoformation" : lTransformation,
               "Trend" : self.mTrend.mFormula,
               "Cycle" : self.mCycle.mFormula,
               "AR_Model" : self.mAR.mFormula,
               };
        dict1["Model"] = d2;
        dict1["Complexity"] = self.getComplexity()
        lCriterion = self.mTimeInfo.mOptions.mModelSelection_Criterion
        dict1["Model_Selection_Criterion"] = lCriterion
        lPerfs = {}
        lForecastColumn = str(self.mOriginalSignal) + "_Forecast";
        for h in [1, self.mTimeInfo.mHorizon]:
            lHorizonName = lForecastColumn + "_" + str(h);
            lPerfs[h] = self.mForecastPerfs[lHorizonName].to_dict()
        dict1["Model_Performance"] = lPerfs;
        if(iWithOptions):
            dict1["Options"] = self.mTimeInfo.mOptions.__dict__
        return dict1;


    def getForecastDatasetForPlots(self):
        lInput = self.mTrend.mSignalFrame;
        lOutput = self.forecast(lInput ,  self.mTimeInfo.mHorizon);
        return lOutput

    def plotResidues(self, name = None, format = 'png', iOutputDF = None):
        df = iOutputDF
        if(df is None):
            df = self.getForecastDatasetForPlots();
        lTime = self.mTimeInfo.mTime; # NormalizedTimeColumn;
        lPrefix = self.mSignal + "_";
        lPrefix2 = str(self.mOriginalSignal) + "_";
        if(name is not None):
            tsplot.decomp_plot(df, lTime, self.mSignal, lPrefix + 'Trend' , lPrefix + 'Trend_residue', name = name + "_Trend" , format=format, horizon = self.mTimeInfo.mHorizon, title = "Trend\n" + self.mTrend.mOutName);
            tsplot.decomp_plot(df, lTime, lPrefix + 'Trend_residue' , lPrefix + 'Cycle', lPrefix + 'Cycle_residue', name = name + "_Cycle" , format=format, horizon = self.mTimeInfo.mHorizon, title = "Cycle\n" + self.mCycle.mOutName);
            tsplot.decomp_plot(df, lTime, lPrefix + 'Cycle_residue' , lPrefix + 'AR' , lPrefix + 'AR_residue', name = name + "_AR" , format=format, horizon = self.mTimeInfo.mHorizon, title = "AR\n" + self.mAR.mOutName);
            tsplot.decomp_plot(df, lTime, self.mSignal, lPrefix + 'TransformedForecast' , lPrefix + 'TransformedResidue', name = name + "_TransformedForecast" , format=format, horizon = self.mTimeInfo.mHorizon, title = "Transformed Signal Forecast\n" + self.getFormula());
            tsplot.decomp_plot(df, lTime, self.mOriginalSignal, lPrefix2 + 'Forecast' , lPrefix2 + 'Residue', name = name + "_Forecast" , format=format, horizon = self.mTimeInfo.mHorizon, title = "Signal Forecast\n" + self.mOutName);
        else:
            tsplot.decomp_plot(df, lTime, self.mSignal, lPrefix + 'Trend' , lPrefix + 'Trend_residue', horizon = self.mTimeInfo.mHorizon, title = "Trend\n" + self.mTrend.mOutName);
            tsplot.decomp_plot(df, lTime, lPrefix + 'Trend_residue' , lPrefix + 'Cycle', lPrefix + 'Cycle_residue', horizon = self.mTimeInfo.mHorizon, title = "Cycle\n" + self.mCycle.mOutName);
            tsplot.decomp_plot(df, lTime, lPrefix + 'Cycle_residue' , lPrefix + 'AR' , lPrefix + 'AR_residue', horizon = self.mTimeInfo.mHorizon, title = "AR\n" + self.mAR.mOutName);
            tsplot.decomp_plot(df, lTime, self.mSignal, lPrefix + 'TransformedForecast' , lPrefix + 'TransformedResidue', horizon = self.mTimeInfo.mHorizon, title = "Transformed Signal Forecast\n" + self.getFormula());
            tsplot.decomp_plot(df, lTime, self.mOriginalSignal, lPrefix2 + 'Forecast' , lPrefix2 + 'Residue', horizon = self.mTimeInfo.mHorizon, title = "Signal Forecast\n" + self.mOutName);

    def get_title_details_for_plots(self, iPrefix):
        lTitle = iPrefix + "\nModel = " + self.mOutName + "\n"
        lCriterion = self.mTimeInfo.mOptions.mModelSelection_Criterion
        lForecastColumn = str(self.mOriginalSignal) + "_Forecast";
        for h in [1, self.mTimeInfo.mHorizon]:
            lHorizonName = lForecastColumn + "_" + str(h);
            lPerf = self.mForecastPerfs[lHorizonName]
            lTitle = lTitle + "MAPE_" + str(h) + " = " + str(lPerf.mMAPE) + " "
            if(lCriterion != "MAPE"):
                lTitle = lTitle + lCriterion + "_" + str(h) + " = " + str(lPerf.getCriterionValue(lCriterion)) + " "
        return lTitle
            
    def standardPlots(self, name = None, format = 'png'):
        lOutput =  self.getForecastDatasetForPlots();
        self.plotResidues(name = name, format=format, iOutputDF = lOutput);
        # print(lOutput.columns)
        lPrefix = str(self.mOriginalSignal) + "_";
        lForecastColumn = lPrefix + 'Forecast';
        lTime = self.mTimeInfo.mTime;            
        lOutput.set_index(lTime, inplace=True, drop=False);
        # print(lOutput[lTime].dtype);

        # Add more informative title for this plot.  Investigate Model Esthetics for PyAF #212 
        lTitle = self.get_title_details_for_plots("Prediction Intervals")
        tsplot.prediction_interval_plot(lOutput,
                                        lTime, self.mOriginalSignal,
                                        lForecastColumn,
                                        lForecastColumn + '_Lower_Bound',
                                        lForecastColumn + '_Upper_Bound',
                                        name = name,
                                        format= format, horizon = self.mTimeInfo.mHorizon,
                                        title = lTitle);
        
        if(self.mTimeInfo.mOptions.mAddPredictionIntervals):
            lQuantiles = self.mPredictionIntervalsEstimator.mForecastPerformances[lForecastColumn + "_1"].mErrorQuantiles.keys()
            lQuantiles = sorted(lQuantiles)
            tsplot.quantiles_plot(lOutput,
                                  lTime, self.mOriginalSignal,
                                  lForecastColumn  ,
                                  lQuantiles,
                                  name = name,
                                  format= format,
                                  horizon = self.mTimeInfo.mHorizon,
                                  title = "Forecast_Quantiles\n" + self.mOutName);
        #lOutput.plot()

    
        
    def getPlotsAsDict(self):
        lDict = {};
        df = self.getForecastDatasetForPlots();
        lTime = self.mTime;
        lSignalColumn = self.mOriginalSignal;
        lPrefix = self.mSignal + "_";
        lPrefix2 = str(self.mOriginalSignal) + "_";
        lDict["Trend"] = tsplot.decomp_plot_as_png_base64(df, lTime, self.mSignal, lPrefix + 'Trend' , lPrefix + 'Trend_residue', name = "trend", horizon = self.mTimeInfo.mHorizon, title = "Trend\n" + self.mTrend.mOutName);
        lDict["Cycle"] = tsplot.decomp_plot_as_png_base64(df, lTime, lPrefix + 'Trend_residue' , lPrefix + 'Cycle', lPrefix + 'Cycle_residue', name = "cycle", horizon = self.mTimeInfo.mHorizon, title = "Cycle\n" + self.mCycle.mOutName);
        lDict["AR"] = tsplot.decomp_plot_as_png_base64(df, lTime, lPrefix + 'Cycle_residue' , lPrefix + 'AR' , lPrefix + 'AR_residue', name = "AR", horizon = self.mTimeInfo.mHorizon, title = "AR\n" + self.mAR.mOutName);
        lDict["TransformedForecast"] = tsplot.decomp_plot_as_png_base64(df, lTime, self.mSignal, lPrefix + 'TransformedForecast' , lPrefix + 'TransformedResidue', name = "TransformedForecast", horizon = self.mTimeInfo.mHorizon, title = "Transformed Signal Forecast\n" + self.getFormula());
        
        lDict["Forecast"] = tsplot.decomp_plot_as_png_base64(df, lTime, lSignalColumn, lPrefix2 + 'Forecast' , lPrefix2 + 'Residue', name = "forecast", horizon = self.mTimeInfo.mHorizon, title = "Signal Forecast\n" + self.mOutName);

        lDict["Prediction_Intervals"] = self.getPredictionIntervalPlot(df)
        if(self.mTimeInfo.mOptions.mAddPredictionIntervals):            
            lDict["Forecast_Quantiles"] = self.getForecastQuantilesPlot(df)
        return lDict;

    def getPredictionIntervalPlot(self, df = None):        
        lOutput = df if df is not None else self.getForecastDatasetForPlots();
        # print(lOutput.columns)
        lPrefix = str(self.mOriginalSignal) + "_";
        lForecastColumn = lPrefix + 'Forecast';
        lTime = self.mTimeInfo.mTime;
        lOutput.set_index(lTime, inplace=True, drop=False);
        # Add more informative title for this plot.  Investigate Model Esthetics for PyAF #212 
        lTitle = self.get_title_details_for_plots("Prediction Intervals")
        return tsplot.prediction_interval_plot_as_png_base64(lOutput,
                                                             lTime, self.mOriginalSignal,
                                                             lForecastColumn  ,
                                                             lForecastColumn + '_Lower_Bound',
                                                             lForecastColumn + '_Upper_Bound',
                                                             name = "prediction_intervals",
                                                             horizon = self.mTimeInfo.mHorizon,
                                                             title = lTitle);
    
    def getForecastQuantilesPlot(self, df = None):
        lOutput = df if df is not None else self.getForecastDatasetForPlots();
        lPrefix = self.mOriginalSignal + "_";
        lTime = self.mTime;
        lForecastColumn = lPrefix + 'Forecast';
        lQuantiles = self.mPredictionIntervalsEstimator.mForecastPerformances[lForecastColumn + "_1"].mErrorQuantiles.keys()
        lQuantiles = sorted(lQuantiles)
        return tsplot.quantiles_plot_as_png_base64(lOutput,
                                                   lTime, self.mOriginalSignal,
                                                   lForecastColumn  ,
                                                   lQuantiles,
                                                   name = "Forecast_Quantiles [" + self.mOutName + "]",
                                                   format= format,
                                                   horizon = self.mTimeInfo.mHorizon,
                                                   title = "Forecast_Quantiles\n" + self.mOutName);

    def getVersions(self):
        lVersionDict = tsutil.getVersions();
        return lVersionDict;

    def clean_dataframes(self):
        self.mTrend.mSignalFrame = self.mTrend.mSignalFrame[[self.mTime, self.mOriginalSignal, self.mSignal]].copy()
        # print(self.mTrend.mSignalFrame.columns)
        
