# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np

from . import PredictionIntervals as predint

from . import Plots as tsplot
from . import Perf as tsperf

class cTimeSeriesModel:
    
    def __init__(self, transf, trend, cycle, autoreg):
        self.mTransformation = transf;
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

    def info(self):
        lSignal = self.mTrend.mSignalFrame[self.mSignal];
        lStr1 = "SignalVariable='" + self.mSignal +"'";
        lStr1 += " Min=" + str(np.min(lSignal)) + " Max="  + str(np.max(lSignal)) + " ";
        lStr1 += " Mean=" + str(np.mean(lSignal)) + " StdDev="  + str(np.std(lSignal));
        return lStr1;

        
    def getComplexity(self):
        lComplexity = 32 * self.mTransformation.mComplexity +  16 * self.mTrend.mComplexity + 4 * self.mCycle.mComplexity + 1 * self.mAR.mComplexity;
        return lComplexity;     

    def updatePerfs(self):
        self.mModelFrame = pd.DataFrame();
        lSignal = self.mTrend.mSignalFrame[self.mSignal]
        N = lSignal.shape[0];
        self.mTrend.mTimeInfo.addVars(self.mModelFrame);
        df = self.forecastOneStepAhead(self.mModelFrame);
        self.mModelFrame = df.head(N);
        # print(self.mModelFrame.columns);
        lPrefix = self.mSignal + "_";
        lFitPerf = tsperf.cPerf();
        lForecastPerf = tsperf.cPerf();
        lTestPerf = tsperf.cPerf();
        (lFrameFit, lFrameForecast, lFrameTest) = self.mTrend.mTimeInfo.cutFrame(self.mModelFrame);
        lFitPerf.compute(lFrameFit[self.mOriginalSignal] , lFrameFit[lPrefix + 'TransformedForecast'] , 'TransformedForecast')
        lForecastPerf.compute(lFrameForecast[self.mOriginalSignal] , lFrameForecast[lPrefix + 'TransformedForecast'], 'TransformedForecast')
        lTestPerf.compute(lFrameTest[self.mOriginalSignal] , lFrameTest[lPrefix + 'TransformedForecast'], 'TransformedForecast')
        self.mFitPerf = lFitPerf
        self.mForecastPerf = lForecastPerf;
        self.mTestPerf = lTestPerf;
        # print("PERF_COMPUTATION" , self.mOutName, self.mFitPerf.mMAPE);
        # self.computePredictionIntervals();
        
    def computePredictionIntervals(self):
        # prediction intervals
        self.mPredictionIntervalsEstimator = predint.cPredictionIntervalsEstimator();
        self.mPredictionIntervalsEstimator.mSignalFrame = self.mTrend.mSignalFrame.copy();
        self.mPredictionIntervalsEstimator.mTime = self.mTime;
        self.mPredictionIntervalsEstimator.mSignal = self.mOriginalSignal;
        self.mPredictionIntervalsEstimator.mHorizon = self.mTimeInfo.mHorizon;
        self.mPredictionIntervalsEstimator.mModel = self;
        
        self.mPredictionIntervalsEstimator.computePerformances();

    def getFormula(self):
        lFormula = self.mTrend.mFormula + " + ";
        lFormula += self.mCycle.mFormula + " + ";
        lFormula += self.mAR.mFormula;
        return lFormula;


    def getInfo(self):
        print("TIME_DETAIL " + self.mTrend.mTimeInfo.info());
        print("SIGNAL_DETAIL " + self.info());
        print("BEST_TRANSOFORMATION_TYPE '" + self.mTransformation.get_name("") + "'");
        print("BEST_DECOMPOSITION  '" + self.mOutName + "' [" + self.getFormula() + "]");
        print("TREND_DETAIL '" + self.mTrend.mOutName + "' [" + self.mTrend.mFormula + "]");
        print("CYCLE_DETAIL '"+ self.mCycle.mOutName + "' [" + self.mCycle.mFormula + "]");
        print("AUTOREG_DETAIL '" + self.mAR.mOutName + "' [" + self.mAR.mFormula + "]");
        print("MODEL_MAPE MAPE_Fit=" + str(self.mFitPerf.mMAPE) + " MAPE_Forecast=" + str(self.mForecastPerf.mMAPE)  + " MAPE_Test=" + str(self.mTestPerf.mMAPE) );
        print("MODEL_L2 L2_Fit=" + str(self.mFitPerf.mL2) + " L2_Forecast=" + str(self.mForecastPerf.mL2)  + " L2_Test=" + str(self.mTestPerf.mL2) );
        print("MODEL_COMPLEXITY ", str(self.getComplexity()) );
        print("AR_MODEL_DETAIL_START");
        # self.mAR.dumpCoefficients();
        print("AR_MODEL_DETAIL_END");



    def forecastOneStepAhead(self , df):
        assert(self.mTime in df.columns)
        assert(self.mOriginalSignal in df.columns)
        lPrefix = self.mSignal + "_";
        df1 = df;
        # df1.to_csv("before.csv");
        # add new line with new time value, row_number and nromalized time
        df1 = self.mTimeInfo.transformDataset(df1);
        # df1.to_csv("after_time.csv");
        # print("TimeInfo update : " , df1.columns);
        # add signal tranformed column
        df1 = self.mTransformation.transformDataset(df1, self.mOriginalSignal);
        # df1.to_csv("after_transformation.csv");
        #print("Transformation update : " , df1.columns);
        # compute the trend based on the transformed column and compute trend residue
        df1 = self.mTrend.transformDataset(df1);
        #print("Trend update : " , df1.columns);
        # df1.to_csv("after_trend.csv");
        # compute the cycle and its residue based on the trend residue
        df1 = self.mCycle.transformDataset(df1);
        # df1.to_csv("after_cycle.csv");
        #print("Cycle update : " , df1.columns);
        # compute the AR componnet and its residue based on the cycle residue
        df1 = self.mAR.transformDataset(df1);
        # df1.to_csv("after_ar.csv");
        #print("AR update : " , df1.columns);
        # compute the forecast and its residue (forecast = trend  + cycle + AR)
        df2 = df1;
        lTrendColumn = df2[self.mTrend.mOutName]
        lCycleColumn = df2[self.mCycle.mOutName]
        lARColumn = df2[self.mAR.mOutName]
        lSignal = df2[self.mSignal]
        df2[lPrefix + 'Trend'] =  lTrendColumn;
        df2[lPrefix + 'Trend_residue'] =  lSignal - lTrendColumn;
        df2[lPrefix + 'Cycle'] =  lCycleColumn;
        df2[lPrefix + 'Cycle_residue'] = df2[lPrefix + 'Trend_residue'] - lCycleColumn;
        df2[lPrefix + 'AR'] =  lARColumn ;
        df2[lPrefix + 'AR_residue'] = df2[lPrefix + 'Cycle_residue'] - lARColumn;
        df2[lPrefix + 'TransformedForecast'] =  lTrendColumn + lCycleColumn + lARColumn ;
        df2[lPrefix + 'TransformedResidue'] =  lSignal - df2[lPrefix + 'TransformedForecast']

        lPrefix2 = self.mOriginalSignal + "_";
        df2[lPrefix2 + 'Forecast'] = self.mTransformation.invert(df2[lPrefix + 'TransformedForecast']);
        lOriginalSignal = df2[self.mOriginalSignal]
        df2[lPrefix2 + 'Residue'] =  lOriginalSignal - df2[lPrefix2 + 'Forecast']
        df2.reset_index(drop=True, inplace=True);
        return df2;


    def forecast(self , df , iHorizon):
        N0 = df.shape[0];
        df1 = self.forecastOneStepAhead(df)
        lForecastColumnName = self.mOriginalSignal + "_Forecast";
        for h in range(0 , iHorizon - 1):
            # print(df1.info());
            N = df1.shape[0];
            # replace the signal with the forecast in the last line  of the dataset
            lPos = df1.index[N - 1];
            lSignal = df1.loc[lPos , lForecastColumnName];
            df1.loc[lPos , self.mOriginalSignal] = lSignal;
            df1 = df1[[self.mTime , self.mOriginalSignal]];
            df1 = self.forecastOneStepAhead(df1)

        assert((N0 + iHorizon) == df1.shape[0])
        N1 = df1.shape[0];
        for h in range(0 , iHorizon):
            df1.loc[N1 - 1 - h, self.mOriginalSignal] = np.nan;
            pass
        df1 = self.addPredictionIntervals(df, df1);
        return df1

    def addPredictionIntervals(self, iInputDS, iForecastFrame):
        lSignalColumn = self.mOriginalSignal;
        lLowerBound = iForecastFrame[lSignalColumn].apply(lambda x : np.nan)
        lUpperBound = lLowerBound.copy();

        N = iInputDS.shape[0];
        lForecastColumn = lSignalColumn + "_Forecast";
        lConfidence = 2.0 ; # 0.95
        # the prediction intervals are only computed for the training horizon
        for h in range(0 , self.mTimeInfo.mHorizon):
            lHorizonName = lForecastColumn + "_" + str(h + 1);
            lWidth = lConfidence * self.mPredictionIntervalsEstimator.mForecastPerformances[lHorizonName].mL2;
            lLowerBound.loc[N + h ] = iForecastFrame.loc[N + h , lForecastColumn] - lWidth;
            lUpperBound.loc[N + h ] = iForecastFrame.loc[N + h , lForecastColumn] + lWidth;
            
        iForecastFrame[lForecastColumn + '_Lower_Bound'] = lLowerBound; 
        iForecastFrame[lForecastColumn + '_Upper_Bound'] = lUpperBound; 
        return iForecastFrame;


    def plotForecasts(self, df):
        lPrefix = self.mSignal + "_";
        lTime = self.mTimeInfo.mNormalizedTimeColumn;            
        tsplot.decomp_plot(df,
                           lTime, lPrefix + 'Signal',
                           lPrefix + 'Forecast' , lPrefix + 'Residue');

    def to_json(self):
        dict1 = {};
        dict1["Dataset"] = { "Time" : self.mTimeInfo.to_json(),
                             "Signal" : self.mOriginalSignal,
                             "Training_Signal_Length" : self.mModelFrame.shape[0]};
        lTransformation = self.mTransformation.mFormula;
        dict1["Model"] = { "Best_Decomposition" : self.mOutName,
                           "Signal_Transoformation" : lTransformation,
                           "Trend" : self.mTrend.mFormula,
                           "Cycle" : self.mCycle.mFormula,
                           "AR_Model" : self.mAR.mFormula,
                           };
        dict1["Model_Performance"] = {"MAPE" : str(self.mForecastPerf.mMAPE),
                                      "RMSE" : str(self.mForecastPerf.mL2),
                                      "COMPLEXITY" : str(self.getComplexity())};
        
        return dict1;

    def plotResidues(self, name = None):
        df = self.mModelFrame;
        lTime = self.mTimeInfo.mTime; # NormalizedTimeColumn;
        lPrefix = self.mSignal + "_";
        if(name is not None):
            tsplot.decomp_plot(df, lTime, self.mSignal, lPrefix + 'Trend' , lPrefix + 'Trend_residue', name = name + "_trend");
            tsplot.decomp_plot(df, lTime, lPrefix + 'Trend_residue' , lPrefix + 'Cycle', lPrefix + 'Cycle_residue', name = name + "_cycle");
            tsplot.decomp_plot(df, lTime, lPrefix + 'Cycle_residue' , lPrefix + 'AR' , lPrefix + 'AR_residue', name = name + "_AR");
            tsplot.decomp_plot(df, lTime, self.mSignal, lPrefix + 'Forecast' , lPrefix + 'Residue', name = name + "_forecast");
        else:
            tsplot.decomp_plot(df, lTime, self.mSignal, lPrefix + 'Trend' , lPrefix + 'Trend_residue');
            tsplot.decomp_plot(df, lTime, lPrefix + 'Trend_residue' , lPrefix + 'Cycle', lPrefix + 'Cycle_residue');
            tsplot.decomp_plot(df, lTime, lPrefix + 'Cycle_residue' , lPrefix + 'AR' , lPrefix + 'AR_residue');
            tsplot.decomp_plot(df, lTime, self.mSignal, lPrefix + 'Forecast' , lPrefix + 'Residue');
        
    def standrdPlots(self, name = None):
        self.plotResidues(name = name);
        lInput = self.mTrend.mSignalFrame;
        lOutput = self.forecast(lInput ,  self.mTimeInfo.mHorizon);
        # print(lOutput.columns)
        lPrefix = self.mOriginalSignal + "_";
        lForecastColumn = lPrefix + 'Forecast';
        lTime = self.mTimeInfo.mTime;            
        lOutput.set_index(lTime, inplace=True, drop=False);
        # print(lOutput[lTime].dtype);
        tsplot.prediction_interval_plot(lOutput,
                                        lTime, self.mOriginalSignal,
                                        lForecastColumn  ,
                                        lForecastColumn + '_Lower_Bound',
                                        lForecastColumn + '_Upper_Bound',
                                        name = name,
                                        max_length = (16 * self.mTimeInfo.mHorizon));
        #lOutput.plot()
        
    def getPlotsAsDict(self):
        lDict = {};
        df = self.mModelFrame;
        lTime = self.mTime;
        lSignalColumn = self.mSignal;
        lPrefix = lSignalColumn + "_";
        lDict["Trend"] = tsplot.decomp_plot_as_png_base64(df, lTime, lSignalColumn, lPrefix + 'Trend' , lPrefix + 'Trend_residue', name = "trend");
        lDict["Cycle"] = tsplot.decomp_plot_as_png_base64(df, lTime, lPrefix + 'Trend_residue' , lPrefix + 'Cycle', lPrefix + 'Cycle_residue', name = "cycle");
        lDict["AR"] = tsplot.decomp_plot_as_png_base64(df, lTime, lPrefix + 'Cycle_residue' , lPrefix + 'AR' , lPrefix + 'AR_residue', name = "AR");
        lDict["Forecast"] = tsplot.decomp_plot_as_png_base64(df, lTime, lSignalColumn, lPrefix + 'Forecast' , lPrefix + 'Residue', name = "forecast");

        lInput = self.mModelFrame[[self.mTime, self.mSignal]];
        lOutput = self.forecast(lInput ,  self.mTimeInfo.mHorizon);
        # print(lOutput.columns)
        lPrefix = self.mOriginalSignal + "_";
        lForecastColumn = lPrefix + 'Forecast';
        lTime = self.mTimeInfo.mTime;
        lOutput.set_index(lTime, inplace=True, drop=False);
        lDict["Prediction_Intervals"] = tsplot.prediction_interval_plot_as_png_base64(lOutput,
                                                                                      lTime, self.mOriginalSignal,
                                                                                      lForecastColumn  ,
                                                                                      lForecastColumn + '_Lower_Bound',
                                                                                      lForecastColumn + '_Upper_Bound',
                                                                                      name = "prediction_intervals",
                                                                                      max_length = (16 * self.mTimeInfo.mHorizon));
        return lDict;
