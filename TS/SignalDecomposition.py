import pandas as pd
import numpy as np
import datetime
import sys

import multiprocessing as mp

# for timing
import time

import threading
from multiprocessing.dummy import Pool as ThreadPool

from . import Time as tsti
from . import Exogenous as tsexog
from . import Signal_Transformation as tstransf
from . import Perf as tsperf
from . import SignalDecomposition_Trend as tstr
from . import SignalDecomposition_Cycle as tscy
from . import SignalDecomposition_AR as tsar
from . import Options as tsopts
from . import TimeSeriesModel as tsmodel

from sklearn.externals import joblib

class cSignalDecompositionOneTransform:
        
    def __init__(self):
        self.mSignalFrame = pd.DataFrame()
        self.mTime = "time"
        self.mSignal = "AirPassengers"
        self.mTimeInfo = tsti.cTimeInfo();
        self.mTrendEstimator = tstr.cTrendEstimator()
        self.mCycleEstimator = tscy.cCycleEstimator();
        self.mAREstimator = tsar.cAutoRegressiveEstimator();
        self.mForecastFrame = pd.DataFrame()
        self.mTransformation = tstransf.cSignalTransform_None();
        

    def computeForecast(self, nextTime):
        trendvalue = computeTrend(self , nextTime);
        cyclevalue = computeCycle(self , nextTime);
        ar_value = computeAR(self , nextTime);
        forecast = trendvalue + cyclevalue + ar_value
        return forecast;

    def serialize(self):
        joblib.dump(self, self.mTimeInfo.mTime + "_" + self.mSignal + "_TS.pkl")        

    def setParams(self , iInputDS, iTime, iSignal, iHorizon, iTransformation, iExogenousData = None):
        assert(iInputDS.shape[0] > 0)
        assert(iInputDS.shape[1] > 0)
        assert(iTime in iInputDS.columns)
        assert(iSignal in iInputDS.columns)

        # print("setParams , head", iInputDS.head());
        # print("setParams , tail", iInputDS.tail());
        # print("setParams , columns", iInputDS.columns);
        
        self.mTime = iTime
        self.mOriginalSignal = iSignal;
        
        self.mTransformation = iTransformation;
        self.mTransformation.mOriginalSignal = iSignal; 
        self.mTransformation.fit(iInputDS[iSignal]);

        self.mSignal = iTransformation.get_name(iSignal)
        self.mHorizon = iHorizon;
        self.mSignalFrame = pd.DataFrame()
#        self.mSignalFrame = iInputDS.copy()
        self.mSignalFrame[self.mOriginalSignal] = iInputDS[iSignal];
        self.mSignalFrame[self.mSignal] = self.mTransformation.apply(iInputDS[iSignal]);
        self.mSignalFrame[self.mTime] = iInputDS[self.mTime].copy();
        self.mSignalFrame['row_number'] = np.arange(0, iInputDS.shape[0]);
        self.mSignalFrame.dropna(inplace = True);
        assert(self.mSignalFrame.shape[0] > 0);

        # print("SIGNAL_INFO " , self.info());
        
        self.mTimeInfo = tsti.cTimeInfo();
        self.mTimeInfo.mTime = self.mTime;
        self.mTimeInfo.mSignal = self.mSignal;
        self.mTimeInfo.mHorizon = self.mHorizon;
        self.mTimeInfo.mSignalFrame = self.mSignalFrame;
        self.mTimeInfo.mOptions = self.mOptions;

        self.mExogenousInfo = None;
        if(iExogenousData is not None):
            self.mExogenousInfo = tsexog.cExogenousInfo();
            self.mExogenousInfo.mExogenousData = iExogenousData;
            self.mExogenousInfo.mDateVariable = self.mTime;
            self.mExogenousInfo.mOptions = self.mOptions;
        
        self.mTrendEstimator.mSignalFrame = self.mSignalFrame
        self.mTrendEstimator.mTimeInfo = self.mTimeInfo
        self.mTrendEstimator.mOptions = self.mOptions;

        self.mCycleEstimator.mTimeInfo = self.mTimeInfo
        self.mCycleEstimator.mOptions = self.mOptions;

        self.mAREstimator.mTimeInfo = self.mTimeInfo
        self.mAREstimator.mExogenousInfo = self.mExogenousInfo;
        self.mAREstimator.mOptions = self.mOptions;


    def collectPerformanceIndices(self) :
        self.mPerfsByModel = {}
        rows_list = [];
        self.mFitPerf = {}
        self.mForecastPerf = {}
        self.mTestPerf = {}
        for trend in self.mAREstimator.mTrendList:
            for cycle in self.mAREstimator.mCycleList[trend]:
                cycle_residue = cycle.getCycleResidueName();
                for autoreg in self.mAREstimator.mARList[cycle_residue]:
                    lModel = tsmodel.cTimeSeriesModel(self.mTransformation, trend, cycle, autoreg);
                    lModel.updatePerfs();
                    lComplexity = lModel.getComplexity();
                    lFitPerf = lModel.mFitPerf;
                    lForecastPerf = lModel.mForecastPerf;
                    lTestPerf = lModel.mTestPerf;
                    self.mPerfsByModel[lModel.mOutName] = [lModel, lComplexity, lFitPerf , lForecastPerf, lTestPerf];
                    row = [lModel.mOutName , lComplexity,
                           lFitPerf.mCount, lFitPerf.mL2, lFitPerf.mMAPE,
                           lForecastPerf.mCount, lForecastPerf.mL2, lForecastPerf.mMAPE,
                           lTestPerf.mCount, lTestPerf.mL2, lTestPerf.mMAPE]
                    rows_list.append(row);
                    if(self.mOptions.mDebugPerformance):
                        print("collectPerformanceIndices : " , row[0] , " ", row[1] , " " , row[7]);
        self.mPerfDetails = pd.DataFrame(rows_list, columns=
                                         ('Model', 'Complexity',
                                          'FitCount', 'FitL2', 'FitMAPE',
                                          'ForecastCount', 'ForecastL2', 'ForecastMAPE',
                                          'TestCount', 'TestL2', 'TestMAPE')) 
        self.mPerfDetails.sort_values(by=['ForecastMAPE' , 'Complexity'] , inplace=True);
        lBestName = self.mPerfDetails.iloc[0]['Model'];
        self.mBestModel = self.mPerfsByModel[lBestName][0];
        return self.mBestModel;
    

    
    def train(self , iInputDS, iTime, iSignal,
              iHorizon, iTransformation):
        self.setParams(iInputDS, iTime, iSignal, iHorizon, iTransformation, self.mExogenousData);
        # estimate time info
        # assert(self.mTimeInfo.mSignalFrame.shape[0] == iInputDS.shape[0])
        self.mTimeInfo.estimate();
        # print("TIME_INFO " , self.mTimeInfo.info());
        self.mSignalFrame[self.mTimeInfo.mNormalizedTimeColumn] = self.mTimeInfo.mSignalFrame[self.mTimeInfo.mNormalizedTimeColumn]
        if(self.mOptions.mEnablePlots):    
            self.plotSignal()

        if(self.mExogenousInfo is not None):
            self.mExogenousInfo.fit();
        # estimate the trend
        self.mTrendEstimator.estimateTrend();
        #self.mTrendEstimator.plotTrend();
        # estimate cycles
        self.mCycleEstimator.mTrendFrame = self.mTrendEstimator.mTrendFrame;
        self.mCycleEstimator.mTrendList = self.mTrendEstimator.mTrendList;
        self.mCycleEstimator.estimateAllCycles();
        # if(self.mOptions.mDebugCycles):
            # self.mCycleEstimator.plotCycles();
        # autoregressive
        self.mAREstimator.mCycleFrame = self.mCycleEstimator.mCycleFrame;
        self.mAREstimator.mTrendList = self.mCycleEstimator.mTrendList;
        self.mAREstimator.mCycleList = self.mCycleEstimator.mCycleList;
        self.mAREstimator.estimate();
        #self.mAREstimator.plotAR();
        # forecast perfs
        self.collectPerformanceIndices()        
        



def run_transform_thread(iInputDS, iTime, iSignal, iHorizon, transform1, iOptions, iExogenousData):
    sigdec = cSignalDecompositionOneTransform();
    sigdec.mOptions = iOptions;
    sigdec.mExogenousData = iExogenousData;
    sigdec.train(iInputDS, iTime, iSignal, iHorizon, transform1);    

class cSignalDecomposition:
        
    def __init__(self):
        self.mSigDecByTransform = {};
        self.mOptions = tsopts.cSignalDecomposition_Options();
        self.mExogenousData = None;
        pass

    def needQuantile(self, df , q):
        N = df.shape[0];
        if(N < (12 * q)) :
            return False;
        return False;

    def validateTransformation(self , transf):
        lName = transf.get_name("");
        # print("Adding Transformation " , lName);
        self.mTransformList = self.mTransformList + [transf];
    
    def defineTransformations(self , df):
        self.mTransformList = [];
        self.validateTransformation(tstransf.cSignalTransform_None());

        if(self.mOptions.mEnableDifferentiationTransforms):
            self.validateTransformation(tstransf.cSignalTransform_Differencing());
            self.validateTransformation(tstransf.cSignalTransform_RelativeDifferencing());
            
        if(self.mOptions.mEnableIntegrationTransforms):
            self.validateTransformation(tstransf.cSignalTransform_Accumulate());

        if(self.mOptions.mEnableCoxBox):
            for i in self.mOptions.mCoxBoxOrders:
                self.validateTransformation(tstransf.cSignalTransform_BoxCox(i));

        if(self.mOptions.mEnableQuantization):
            for q in self.mOptions.mQuantiles:
                if(self.needQuantile(df , q)):
                    self.validateTransformation(tstransf.cSignalTransform_Quantize(q));
        

        for transform1 in self.mTransformList:
            transform1.mOptions = self.mOptions;
            # transform1.test();

            
    def train_threaded(self , iInputDS, iTime, iSignal, iHorizon):
        threads = [] 
        self.defineTransformations();
        for transform1 in self.mTransformList:
            t = threading.Thread(target=run_transform_thread,
                                 args = (iInputDS, iTime, iSignal, iHorizon, transform1, self.mOptions, self.mExogenousData))
            t.daemon = False
            threads += [t] 
            t.start()
 
        for t in threads: 
            t.join()
        
    def train_multiprocessed(self , iInputDS, iTime, iSignal, iHorizon):
        pool = mp.Pool()
        self.defineTransformations();
        for transform1 in self.mTransformList:
            args = (iInputDS, iTime, iSignal, iHorizon,
                    transform1, self.mOptions, self.mExogenousData);
            asyncResult = pool.map_async(run_transform_thread, args);

        resultList = asyncResult.get()
	
        
    def train_not_threaded(self , iInputDS, iTime, iSignal, iHorizon):
        self.defineTransformations(iInputDS);
        for transform1 in self.mTransformList:
            sigdec = cSignalDecompositionOneTransform();
            sigdec.mOptions = self.mOptions;
            sigdec.mExogenousData = self.mExogenousData;
            sigdec.train(iInputDS, iTime, iSignal, iHorizon, transform1);
            self.mSigDecByTransform[transform1.get_name("")] = sigdec


    def collectPerformanceIndices(self) :
        rows_list = []
        self.mPerfsByModel = {}
        for transform1 in self.mTransformList:
            sigdec = self.mSigDecByTransform[transform1.get_name("")]
            for (model , value) in sigdec.mPerfsByModel.items():
                self.mPerfsByModel[model] = value;
                lTranformName = sigdec.mSignal;
                lModelFormula = model
                #  value format : self.mPerfsByModel[lModel.mOutName] = [lModel, lComplexity, lFitPerf , lForecastPerf, lTestPerf];
                lComplexity = value[1];
                lFitPerf = value[2];
                lForecastPerf = value[3];
                lTestPerf = value[4];
                row = [lTranformName, lModelFormula , lComplexity,
                       lFitPerf.mCount, lFitPerf.mL2, lFitPerf.mMAPE,
                       lForecastPerf.mCount, lForecastPerf.mL2, lForecastPerf.mMAPE,
                       lTestPerf.mCount, lTestPerf.mL2, lTestPerf.mMAPE]
                rows_list.append(row);
                if(self.mOptions.mDebugPerformance):
                    print("collectPerformanceIndices : " , row[0] , " ", row[1] , " " , row[2] , " ", row[8]);

        self.mTrPerfDetails =  pd.DataFrame(rows_list, columns=
                                            ('Transformation', 'Model', 'Complexity',
                                             'FitCount', 'FitL2', 'FitMAPE',
                                             'ForecastCount', 'ForecastL2', 'ForecastMAPE',
                                             'TestCount', 'TestL2', 'TestMAPE')) 
        # print(self.mTrPerfDetails.head(self.mTrPerfDetails.shape[0]));
        lBestPerf = self.mTrPerfDetails['ForecastMAPE'].min();
        # allow a loss of one point (0.01 of MAPE) if complexity is reduced.
        self.mTrPerfDetails.sort_values(by=['ForecastMAPE', 'Complexity'] , inplace=True)
        lInterestingModels = self.mTrPerfDetails[self.mTrPerfDetails['ForecastMAPE'] <= (lBestPerf + 0.01)].copy();
        lInterestingModels.sort_values(by=['Complexity'] , inplace=True)        
        lBestName = lInterestingModels.iloc[0]['Model'];
        self.mBestModel = self.mPerfsByModel[lBestName][0];

    def train(self , iInputDS, iTime, iSignal, iHorizon, iExogenousData = None):
        print("START_TRAINING '" + iSignal + "'")
        start_time = time.time()

        self.mTrainingDataset = iInputDS; 
        self.mExogenousData = iExogenousData;
        
        if(self.mOptions.mParallelMode):
            self.train_multiprocessed(iInputDS, iTime, iSignal, iHorizon);
        else:
            self.train_not_threaded(iInputDS, iTime, iSignal, iHorizon);
    
        self.collectPerformanceIndices();

        # Prediction Intervals
        self.mBestModel.computePredictionIntervals();

        if(self.mOptions.mEnablePlots):    
            self.mBestModel.plotForecasts();

        end_time = time.time()
        self.mTrainingTime = end_time - start_time;
        print("END_TRAINING_TIME_IN_SECONDS '" + iSignal + "' " + str(self.mTrainingTime))
        pass

    def forecast(self , iInputDS, iHorizon):
        lForecastFrame = self.mBestModel.forecast(iInputDS, iHorizon);
        return lForecastFrame;


    def getModelFormula(self):
        lFormula = self.mBestModel.getFormula();
        return lFormula;


    def getModelInfo(self):
        return self.mBestModel.getInfo();

    def to_json(self):
        dict1 = self.mBestModel.to_json();
        return dict1;
        
    def standrdPlots(self, name = None):
        self.mBestModel.standrdPlots(name);
        
    def getPlotsAsDict(self):
        lDict = self.mBestModel.getPlotsAsDict();
        return lDict;
