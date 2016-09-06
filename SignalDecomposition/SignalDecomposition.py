import pandas as pd
import numpy as np
import datetime
import sys

import multiprocessing as mp

# for timing
import time

import threading
from multiprocessing.dummy import Pool as ThreadPool

from . import SignalDecomposition_Time as tsti
from . import SignalDecomposition_Signal as tssig
from . import SignalDecomposition_Perf as tsperf
from . import SignalDecomposition_Trend as tstr
from . import SignalDecomposition_Cycle as tscy
from . import SignalDecomposition_AR as tsar
from . import SignalDecomposition_PredictionIntervals as predint
from . import SignalDecomposition_Plots as tsplot
from . import SignalDecomposition_Options as tsopts

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
        self.mTransformation = tssig.cSignalTransform_None();
        

    def info(self):
        lSignal = self.mSignalFrame[self.mSignal];
        lStr1 = "SignalVariable='" + self.mSignal +"'";
        lStr1 += " Min=" + str(np.min(lSignal)) + " Max="  + str(np.max(lSignal)) + " ";
        lStr1 += " Mean=" + str(np.mean(lSignal)) + " StdDev="  + str(np.std(lSignal));
        return lStr1;

    def computeForecast(self, nextTime):
        trendvalue = computeTrend(self , nextTime);
        cyclevalue = computeCycle(self , nextTime);
        ar_value = computeAR(self , nextTime);
        forecast = trendvalue + cyclevalue + ar_value
        return forecast;

    def serialize(self):
        joblib.dump(self, self.mTimeInfo.mTime + "_" + self.mSignal + "_TS.pkl")        

    def setParams(self , iInputDS, iTime, iSignal, iHorizon, iTransformation):
        assert(iInputDS.shape[0] > 0)
        assert(iInputDS.shape[1] > 0)
        assert(iTime in iInputDS.columns)
        assert(iSignal in iInputDS.columns)

        self.mTime = iTime
        self.mOriginalSignal = iSignal;
        
        self.mTransformation = iTransformation;
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

        print("SIGNAL_INFO " , self.info());
        
        self.mTimeInfo = tsti.cTimeInfo();
        self.mTimeInfo.mTime = self.mTime;
        self.mTimeInfo.mSignal = self.mSignal;
        self.mTimeInfo.mHorizon = self.mHorizon;
        self.mTimeInfo.mSignalFrame = self.mSignalFrame;
        self.mTimeInfo.mOptions = self.mOptions;
        
        self.mTrendEstimator.mSignalFrame = self.mSignalFrame
        self.mTrendEstimator.mTimeInfo = self.mTimeInfo
        self.mTrendEstimator.mOptions = self.mOptions;

        self.mCycleEstimator.mTimeInfo = self.mTimeInfo
        self.mCycleEstimator.mOptions = self.mOptions;

        self.mAREstimator.mTimeInfo = self.mTimeInfo
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
                    df = pd.DataFrame();
                    df['Signal'] = self.mSignalFrame[self.mOriginalSignal]
                    df['Model'] = df['Signal'] - autoreg.mARFrame[autoreg.mOutName + "_residue"]
                    df['Model'] = self.mTransformation.invert(df['Model']);
                    lFitPerf = tsperf.cPerf();
                    lForecastPerf = tsperf.cPerf();
                    lTestPerf = tsperf.cPerf();
                    (lFrameFit, lFrameForecast, lFrameTest) = self.mTimeInfo.cutFrame(df);
                    lFitPerf.compute(lFrameFit['Signal'] , lFrameFit['Model'] , 'Model')
                    lForecastPerf.compute(lFrameForecast['Signal'] , lFrameForecast['Model'], 'Model')
                    lTestPerf.compute(lFrameTest['Signal'] , lFrameTest['Model'], 'Model')
                    self.mFitPerf[autoreg] = lFitPerf
                    self.mForecastPerf[autoreg] = lForecastPerf;
                    self.mTestPerf[autoreg] = lTestPerf;
                    self.mPerfsByModel[autoreg.mOutName] = [lFitPerf , lForecastPerf, lTestPerf];
                    row = [autoreg.mOutName ,
                           lFitPerf.mCount, lFitPerf.mL2, lFitPerf.mMAPE,
                           lForecastPerf.mCount, lForecastPerf.mL2, lForecastPerf.mMAPE,
                           lTestPerf.mCount, lTestPerf.mL2, lTestPerf.mMAPE]
                    rows_list.append(row);
        df = pd.DataFrame(rows_list, columns=
                          ('Model',
                           'FitCount', 'FitL2', 'FitMAPE',
                           'ForecastCount', 'ForecastL2', 'ForecastMAPE',
                           'TestCount', 'TestL2', 'TestMAPE')) 
        return df;

    def plotModel(self, df , name = None):
        lTime = self.mTimeInfo.mNormalizedTimeColumn;
        lPrefix = self.mSignal + "_BestModel";
        tsplot.decomp_plot(df, lTime, self.mSignal, lPrefix + 'Trend' , lPrefix + 'Trend_residue', name = name + "_trend");
        tsplot.decomp_plot(df, lTime, lPrefix + 'Trend_residue' , lPrefix + 'Cycle', lPrefix + 'Cycle_residue', name = name + "_cycle");
        tsplot.decomp_plot(df, lTime, lPrefix + 'Cycle_residue' , lPrefix + 'AR' , lPrefix + 'AR_residue', name = name + "_AR");
        tsplot.decomp_plot(df, lTime, self.mSignal, lPrefix + 'Forecast' , lPrefix + 'Residue', name = name + "_forecast");

    def reviewBestModel(self):
        self.mBestModelFrame = pd.DataFrame();
        lSignal = self.mSignalFrame[self.mSignal]
        for trend in self.mAREstimator.mTrendList:
            for cycle in self.mAREstimator.mCycleList[trend]:
                cycle_residue = cycle.getCycleResidueName();
                for autoreg in self.mAREstimator.mARList[cycle_residue]:
                    if(autoreg.mOutName == self.mBestModelName):
                        self.mBestModelTrend = trend;
                        self.mBestModelCycle = cycle
                        self.mBestModelAR = autoreg;
                        self.mBestModelFitPerf = self.mFitPerf[autoreg]
                        self.mBestModelForecastPerf = self.mForecastPerf[autoreg]
                        self.mBestModelFrames = [trend.mTrendFrame, cycle.mCycleFrame, autoreg.mARFrame]
                        self.mTimeInfo.addVars(self.mBestModelFrame);
                        lTrendColumn = trend.mTrendFrame[self.mBestModelTrend.mOutName]
                        lCycleColumn = cycle.mCycleFrame[self.mBestModelCycle.mOutName]
                        lARColumn = autoreg.mARFrame[self.mBestModelAR.mOutName]
                        lPrefix = self.mSignal + "_BestModel";
                        self.mBestModelFrame[lPrefix + 'Trend'] =  lTrendColumn;
                        self.mBestModelFrame[lPrefix + 'Trend_residue'] =  lSignal - lTrendColumn;
                        self.mBestModelFrame[lPrefix + 'Cycle'] =  lCycleColumn;
                        self.mBestModelFrame[lPrefix + 'Cycle_residue'] = self.mBestModelFrame[lPrefix + 'Trend_residue'] - lCycleColumn;
                        self.mBestModelFrame[lPrefix + 'AR'] =  lARColumn ;
                        self.mBestModelFrame[lPrefix + 'AR_residue'] = self.mBestModelFrame[lPrefix + 'Cycle_residue'] - lARColumn;
                        self.mBestModelFrame[lPrefix + 'Forecast'] =  lTrendColumn + lCycleColumn + lARColumn ;
                        self.mBestModelFrame[lPrefix + 'Residue'] =  lSignal - self.mBestModelFrame[lPrefix + 'Forecast']
                        
        if(self.mOptions.mEnablePlots):    
            self.plotModel(self.mBestModelFrame)

    def forecastModelOneStepAhead(self , df):
        assert(self.mTime in df.columns)
        assert(self.mOriginalSignal in df.columns)
        df1 = df.copy();
        # add new line with new time value, row_number and nromalized time
        df1 = self.mTimeInfo.transformDataset(df1);
        #print("TimeInfo update : " , df1.columns);
        # add signal tranformed column
        df1 = self.mTransformation.transformDataset(df1, self.mOriginalSignal);
        #print("Transformation update : " , df1.columns);
        # compute the trend based on the transformed column and compute trend residue
        df1 = self.mBestModelTrend.transformDataset(df1);
        #print("Trend update : " , df1.columns);
        # compute the cycle and its residue based on the trend residue
        df1 = self.mBestModelCycle.transformDataset(df1);
        #print("Cycle update : " , df1.columns);
        # compute the AR componnet and its residue based on the cycle residue
        df1 = self.mBestModelAR.transformDataset(df1);
        #print("AR update : " , df1.columns);
        # compute the forecast and its residue (forecast = trend  + cycle + AR)
        df2 = df1.copy();
        lPrefix = self.mSignal + "_BestModel";
        lTrendColumn = df2[self.mBestModelTrend.mOutName]
        lCycleColumn = df2[self.mBestModelCycle.mOutName]
        lARColumn = df2[self.mBestModelAR.mOutName]
        lSignal = df2[self.mSignal]
        df2[lPrefix + 'Trend'] =  lTrendColumn;
        df2[lPrefix + 'Trend_residue'] =  lSignal - lTrendColumn;
        df2[lPrefix + 'Cycle'] =  lCycleColumn;
        df2[lPrefix + 'Cycle_residue'] = df2[lPrefix + 'Trend_residue'] - lCycleColumn;
        df2[lPrefix + 'AR'] =  lARColumn ;
        df2[lPrefix + 'AR_residue'] = df2[lPrefix + 'Cycle_residue'] - lARColumn;
        df2[lPrefix + 'TransformedForecast'] =  lTrendColumn + lCycleColumn + lARColumn ;
        df2[lPrefix + 'TransformedResidue'] =  lSignal - df2[lPrefix + 'TransformedForecast']

        lPrefix2 = self.mOriginalSignal + "_BestModel";
        df2[lPrefix2 + 'Forecast'] = self.mTransformation.invert(df2[lPrefix + 'TransformedForecast']);
        lOriginalSignal = df2[self.mOriginalSignal]
        df2[lPrefix2 + 'Residue'] =  lOriginalSignal - df2[lPrefix2 + 'Forecast']
        df2.reset_index(drop=True, inplace=True);
        return df2;


    def forecastModel(self , df , iHorizon):
        N0 = df.shape[0];
        df1 = self.forecastModelOneStepAhead(df)
        lForecastColumnName = self.mOriginalSignal + "_BestModelForecast";
        for h in range(0 , iHorizon - 1):
            N = df1.shape[0];
            # replace the signal with the forecast in the last line  of the dataset
            df1.loc[N - 1 , self.mOriginalSignal] = df1.loc[N - 1 , lForecastColumnName];
            df1 = self.forecastModelOneStepAhead(df1)
        assert((N0 + iHorizon) == df1.shape[0])
        N1 = df1.shape[0];
        for h in range(0 , iHorizon):
            df1.loc[N1 - 1 - h, self.mOriginalSignal] = np.nan;
        return df1
    
    def train(self , iInputDS, iTime, iSignal,
              iHorizon, iTransformation):
        self.setParams(iInputDS, iTime, iSignal, iHorizon, iTransformation);
        # estimate time info
        # assert(self.mTimeInfo.mSignalFrame.shape[0] == iInputDS.shape[0])
        self.mTimeInfo.estimate();
        print("TIME_INFO " , self.mTimeInfo.info());
        self.mSignalFrame[self.mTimeInfo.mNormalizedTimeColumn] = self.mTimeInfo.mSignalFrame[self.mTimeInfo.mNormalizedTimeColumn]
        if(self.mOptions.mEnablePlots):    
            self.plotSignal()
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
        self.mPerfDetails = self.collectPerformanceIndices()        
        self.mPerfDetails.sort_values('ForecastMAPE' , inplace=True)
        #print(self.mPerfDetails.head(self.mPerfDetails.shape[0]));
        self.mBestModelName = self.mPerfDetails.iloc[0]['Model']
        self.reviewBestModel();
        # Prediction Intervals
        



def run_transform_thread(iInputDS, iTime, iSignal, iHorizon, transform1, iOptions):
    sigdec = cSignalDecompositionOneTransform();
    sigdec.mOptions = iOptions;
    sigdec.train(iInputDS, iTime, iSignal, iHorizon, transform1);    

class cSignalDecomposition:
        
    def __init__(self):
        self.mSigDecByTranform = {};
        self.mOptions = tsopts.cSignalDecomposition_Options();
        pass

    def needQuantile(self, df , i):
        N = df.shape[0];
        if(N < (12 * i)) :
            return False;
        return True;

    def validateTransformation(self , transf):
        lName = transf.get_name("");
        print("Adding Transformation " , lName);
        self.mTransformList = self.mTransformList + [transf];
    
    def defineTransformations(self , df):
        self.mTransformList = [];
        self.validateTransformation(tssig.cSignalTransform_None());

        if(self.mOptions.mEnableDifferentiationTransforms):
            self.validateTransformation(tssig.cSignalTransform_Differencing());
            self.validateTransformation(tssig.cSignalTransform_RelativeDifferencing());
            
        if(self.mOptions.mEnableIntegrationTransforms):
            self.validateTransformation(tssig.cSignalTransform_Accumulate());

        if(self.mOptions.mEnableCoxBox):
            for i in self.mOptions.mCoxBoxOrders:
                self.validateTransformation(tssig.cSignalTransform_BoxCox(i));

        if(self.mOptions.mEnableQuantization):
            for i in self.mOptions.mQuantiles:
                if(self.needQuantile(df , i)):
                    self.validateTransformation(tssig.cSignalTransform_Quantize(i));
        

        for transform1 in self.mTransformList:
            transform1.mOptions = self.mOptions;
            transform1.test();

            
    def train_threaded(self , iInputDS, iTime, iSignal, iHorizon):
        threads = [] 
        self.defineTransformations();
        for transform1 in self.mTransformList:
            t = threading.Thread(target=run_transform_thread,
                                 args = (iInputDS, iTime, iSignal, iHorizon, transform1, self.mOptions))
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
                    transform1, self.mOptions);
            asyncResult = pool.map_async(run_transform_thread, args);

        resultList = asyncResult.get()
	
        
    def train_not_threaded(self , iInputDS, iTime, iSignal, iHorizon):
        self.mTrainingDataset = iInputDS; 
        self.defineTransformations(iInputDS);
        for transform1 in self.mTransformList:
            sigdec = cSignalDecompositionOneTransform();
            sigdec.mOptions = self.mOptions;
            sigdec.train(iInputDS, iTime, iSignal, iHorizon, transform1);
            self.mSigDecByTranform[transform1.get_name("")] = sigdec

    def createBestModelFrame(self):
        self.mBestModelFrame = pd.DataFrame();
        for transform1 in self.mTransformList:
            sigdec = self.mSigDecByTranform[transform1.get_name("")]
            sigdec.mTimeInfo.addVars(self.mBestModelFrame);

    def plotModelForecasts(self, df):
        for transform1 in self.mTransformList:
            sigdec = self.mSigDecByTranform[transform1.get_name("")]
            lPrefix = sigdec.mSignal + "_BestModel";
            lTime = sigdec.mTimeInfo.mNormalizedTimeColumn;            
            tsplot.decomp_plot(df,
                               lTime, lPrefix + 'Signal',
                               lPrefix + 'Forecast' , lPrefix + 'Residue');

    def collectPerformanceIndices(self) :
        rows_list = []
        self.mFitPerf = {}
        self.mForecastPerf = {}
        self.mTestPerf = {}
        for transform1 in self.mTransformList:
            sigdec = self.mSigDecByTranform[transform1.get_name("")]
            lTranformName = sigdec.mSignal;
            lPrefix = sigdec.mSignal + "_BestModel";
            lSig = self.mBestModelFrame[lPrefix + 'Signal']
            lModelFormula = sigdec.mBestModelName
            lModel = self.mBestModelFrame[lPrefix + 'Forecast']
            lFitPerf = tsperf.cPerf();
            lForecastPerf = tsperf.cPerf();
            lTestPerf = tsperf.cPerf();
            (lFrameFit, lFrameForecast, lFrameTest) = sigdec.mTimeInfo.cutFrame(self.mBestModelFrame);
            lFitPerf.compute(lFrameFit[lPrefix + 'Signal'] , lFrameFit[lPrefix + 'Forecast'] , lPrefix + 'Forecast')
            lForecastPerf.compute(lFrameForecast[lPrefix + 'Signal'] , lFrameForecast[lPrefix + 'Forecast'], lPrefix + 'Forecast')
            lTestPerf.compute(lFrameTest[lPrefix + 'Signal'] , lFrameTest[lPrefix + 'Forecast'], lPrefix + 'Forecast')
            self.mFitPerf[lTranformName] = lFitPerf
            self.mForecastPerf[lTranformName] = lForecastPerf;
            self.mTestPerf[lTranformName] = lTestPerf;
            row = [lTranformName, lModelFormula ,
                   lFitPerf.mCount, lFitPerf.mL2, lFitPerf.mMAPE,
                   lForecastPerf.mCount, lForecastPerf.mL2, lForecastPerf.mMAPE,
                   lTestPerf.mCount, lTestPerf.mL2, lTestPerf.mMAPE]
            rows_list.append(row);

        df = pd.DataFrame(rows_list, columns=
                          ('Transformation', 'Model',
                           'FitCount', 'FitL2', 'FitMAPE',
                           'ForecastCount', 'ForecastL2', 'ForecastMAPE',
                           'TestCount', 'TestL2', 'TestMAPE')) 
        return df;
        
        
    def train(self , iInputDS, iTime, iSignal, iHorizon):
        print("START_TRAINING '" + iSignal + "'")
        start_time = time.time()
        
        if(self.mOptions.mParallelMode):
            self.train_multiprocessed(iInputDS, iTime, iSignal, iHorizon);
        else:
            self.train_not_threaded(iInputDS, iTime, iSignal, iHorizon);
    
        self.createBestModelFrame();
        for transform1 in self.mTransformList:
            sigdec = self.mSigDecByTranform[transform1.get_name("")]
            lPrefix = sigdec.mSignal + "_BestModel";
            self.mBestModelFrame[lPrefix + 'Forecast'] = sigdec.mTransformation.invert(sigdec.mBestModelFrame[lPrefix + 'Forecast']);
            self.mBestModelFrame[lPrefix + 'Signal'] = sigdec.mTransformation.invert(sigdec.mBestModelFrame[sigdec.mSignal]);
            self.mBestModelFrame[lPrefix + 'Residue'] =  self.mBestModelFrame[lPrefix + 'Signal'] - self.mBestModelFrame[lPrefix + 'Forecast'];

        if(self.mOptions.mEnablePlots):    
            self.plotModelForecasts(self.mBestModelFrame);

        self.mTrPerfDetails = self.collectPerformanceIndices();
        self.mTrPerfDetails.sort_values('ForecastMAPE' , inplace=True)
        # print(self.mTrPerfDetails.head(self.mTrPerfDetails.shape[0]));
        self.mBestTransformationName = self.mTrPerfDetails.iloc[0]['Transformation']

        for transform1 in self.mTransformList:
            sigdec = self.mSigDecByTranform[transform1.get_name("")]
            lTranformName = sigdec.mSignal;
            if(lTranformName == self.mBestTransformationName):
                self.mBestTransformation = sigdec;

        # predcition intervals
        self.mPredictionIntervalsEstimator = predint.cPredictionIntervalsEstimator();
        self.mPredictionIntervalsEstimator.mSignalFrame = iInputDS;
        self.mPredictionIntervalsEstimator.mTime = iTime;
        self.mPredictionIntervalsEstimator.mSignal = iSignal;
        self.mPredictionIntervalsEstimator.mHorizon = iHorizon;
        self.mPredictionIntervalsEstimator.mSignalDecomposition = self;
        
        self.mPredictionIntervalsEstimator.computePerformances();
        
        end_time = time.time()
        print("END_TRAINING_TIME_IN_SECONDS '" + iSignal + "' " + str(end_time - start_time))
        pass

    def forecast(self , iInputDS, iHorizon):
        lForecastFrame = self.mBestTransformation.forecastModel(iInputDS, iHorizon);

        lSignalColumn = self.mBestTransformation.mOriginalSignal;
        lLowerBound = lForecastFrame[lSignalColumn].apply(lambda x : np.nan)
        lUpperBound = lLowerBound.copy();

        N = iInputDS.shape[0];
        lForecastColumn = lSignalColumn + "_BestModelForecast";
        lConfidence = 2.0 ; # 0.95
        for h in range(0 , iHorizon):
            lHorizonName = lForecastColumn + "_" + str(h + 1);
            lWidth = lConfidence * self.mPredictionIntervalsEstimator.mForecastPerformances[lHorizonName].mL2;
            lLowerBound.loc[N + h ] = lForecastFrame.loc[N + h , lForecastColumn] - lWidth;
            lUpperBound.loc[N + h ] = lForecastFrame.loc[N + h , lForecastColumn] + lWidth;
            
        lForecastFrame[lForecastColumn + '_Lower_Bound'] = lLowerBound; 
        lForecastFrame[lForecastColumn + '_Upper_Bound'] = lUpperBound; 
        return lForecastFrame;



    def getModelFormula(self):
        lFormula = self.mBestTransformation.mBestModelTrend.mFormula + " + ";
        lFormula += self.mBestTransformation.mBestModelCycle.mFormula + " + ";
        lFormula += self.mBestTransformation.mBestModelAR.mFormula;
        return lFormula;


    def getModelInfo(self):
        print("Time Info : " + self.mBestTransformation.mTimeInfo.info());
        print("Signal Info : " + self.mBestTransformation.info());
        print("Transoformation Info : Type='" + self.mBestTransformation.mTransformation.get_name("") + "'");
        # print("Signal Info : Variable='" + self.mBestTransformation.mOriginalSignal + "'");
        print("Signal length=" + str(self.mBestTransformation.mBestModelFrame.shape[0])) ;
        print("Decomposition Info Model = '" + self.mBestTransformation.mBestModelName + "' [" + self.getModelFormula() + "]");
        print("Trend Info Trend='" + self.mBestTransformation.mBestModelTrend.mOutName + "' [" + self.mBestTransformation.mBestModelTrend.mFormula + "]");
        print("Cycle Info Cycle='"+ self.mBestTransformation.mBestModelCycle.mOutName + "' [" + self.mBestTransformation.mBestModelCycle.mFormula + "]");
        print("AR Info AR='" + self.mBestTransformation.mBestModelAR.mOutName + "' [" + self.mBestTransformation.mBestModelAR.mFormula + "]");
        print("Performance Info MAPE_Fit=" + str(self.mTrPerfDetails.iloc[0].FitMAPE) + " MAPE_Forecast=" + str(self.mTrPerfDetails.iloc[0].ForecastMAPE)  + " MAPE_Test=" + str(self.mTrPerfDetails.iloc[0].TestMAPE) );

    def to_json(self):
        dict1 = {};
        dict1["Dataset"] = { "Time" : self.mBestTransformation.mTimeInfo.to_json(),
                             "Signal" : self.mBestTransformation.mOriginalSignal,
                             "Signal_length" : self.mBestTransformation.mBestModelFrame.shape[0]};
        dict1["Model"] = { "Best_Decomposition" : self.mBestTransformation.mBestModelName,
                           "Signal_Transoformation" : self.mBestTransformation.mTransformation.get_name(""),
                           "Trend" : self.mBestTransformation.mBestModelTrend.mOutName,
                           "Cycle" : self.mBestTransformation.mBestModelCycle.mOutName,
                           "AR_Model" : self.mBestTransformation.mBestModelAR.mOutName};
        dict1["Performance Info By Transformation"] = self.mTrPerfDetails.to_json();
        return dict1;
        
    def standrdPlots(self, name = None):
        self.mBestTransformation.plotModel(self.mBestTransformation.mBestModelFrame, name = name);
        sigdec = self.mBestTransformation;
        lInput = sigdec.mSignalFrame[[sigdec.mTime, sigdec.mSignal]];
        lOutput = self.forecast(lInput ,  sigdec.mHorizon);
        print(lOutput.columns)
        lPrefix = sigdec.mOriginalSignal + "_BestModel";
        lForecastColumn = lPrefix + 'Forecast';
        lTime = sigdec.mTimeInfo.mNormalizedTimeColumn;            
        tsplot.prediction_interval_plot(lOutput,
                                        lTime, sigdec.mOriginalSignal,
                                        lForecastColumn  ,
                                        lForecastColumn + '_Lower_Bound',
                                        lForecastColumn + '_Upper_Bound',
                                        name = name,
                                        max_length = (4 * sigdec.mHorizon));
        #lOutput.plot()
        
        
