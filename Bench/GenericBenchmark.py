import pandas as pd
import numpy as np

import AutoForecast as autof
import TS_CodeGenerator as tscodegen
import Bench.TS_datasets as tsds

import sys,os
# for timing
import time

import multiprocessing as mp
import threading
from multiprocessing.dummy import Pool as ThreadPool



def createDirIfNeeded(dirname):
    try:
        os.mkdir(dirname);
    except:
        pass

class cBenchmarkError(Exception):
    def __init__(self, msg):
        super().__init__(msg);
        pass
    
    
def run_bench_process(a):
    createDirIfNeeded("logs");
    createDirIfNeeded("logs/" + a.mBenchName);    
    logfile = open("logs/" + a.mBenchName + "/PyAutoForecast_" + a.getName()+ ".log", 'w');    
    sys.stdout = logfile    
    try:
        tester = cGeneric_OneSignal_Tester(a.mTSSpec , a.mBenchName);
        tester.testSignal(a.mSignal, a.mHorizon)
        print("BENCHMARK_SUCCESS '" + a.getName() + "'");
        del tester;
        logfile.close();
    except cBenchmarkError as error:
        logger.error(error)
        pass;
    except:
        print("BENCHMARK_FAILURE '" + a.getName() + "'");
        logfile.close();
        raise

class cGeneric_Tester_Arg:
    def __init__(self , bench_name, tsspec, sig, horizon):
        self.mBenchName = bench_name;
        self.mSignal = sig;
        self.mTSSpec = tsspec;
        self.mHorizon = horizon

    def getName(self):
        return self.mBenchName + "_" + self.mSignal + "_" + str(self.mHorizon);


class cGeneric_OneSignal_Tester:

    '''
    '''
        
    def __init__(self , tsspec, bench_name):
        print("BENCH_DATA" , bench_name, tsspec)
        self.mTSSpec = tsspec;
        self.mTrainDataset = {};
        self.mAutoForecastBySignal = {}
        self.mTrainPerfData = {}
        self.mTestPerfData = {}
        self.mTrainTime = {};
        self.mBenchName = bench_name;


    def reportTrainingDataInfo(self, iSignal, iHorizon):
        df = self.mTrainDataset[iSignal  + "_" + str(iHorizon)];
        lDate = 'Date'
        print("TIME : ", lDate  , "N=", df[lDate].shape[0], "H=", iHorizon,
              "HEAD=", df[lDate].head().values, "TAIL=", df[lDate].tail().values);
        print("SIGNAL : ", iSignal , "N=", df[iSignal].shape[0], "H=", iHorizon,
              "HEAD=", df[iSignal].head().values, "TAIL=", df[iSignal].tail().values);
        df.to_csv("bench.csv");
        # print(df.head());

    def checkHorizon(self, N , iHorizon):
        if(N <= iHorizon):
            raise cBenchmarkError('Dataset too short for the requested horizon N=' + str(N) + " H=" + str(iHorizon));


    def getTrainingDataset(self, iSignal, iHorizon):
        df = pd.DataFrame();
        lSignalDataset = self.mTSSpec.mFullDataset;
        lFullDF = lSignalDataset[iSignal].dropna()
        #.astype(np.double)
        N = lFullDF.shape[0]
        # iHorizon = iHorizon; #self.mTSSpec.mHorizon[iSignal]
        lSize = N - iHorizon;
        self.checkHorizon(N , iHorizon);
        df[iSignal] = lFullDF[0: lSize];
        df['Date'] = range(0 , df.shape[0]);
        self.mTrainDataset[iSignal  + "_" + str(iHorizon)] = df;
        self.reportTrainingDataInfo(iSignal, iHorizon);

    def reportModelInfo(self, iModel):
        iModel.getModelInfo();
        print(iModel.mSignalDecomposition.mTrPerfDetails)
        
    def trainModel(self, iSignal, iHorizon):
        df = self.mTrainDataset[iSignal  + "_" + str(iHorizon)];
        lAutoF1 = autof.cAutoForecast();
        self.mAutoForecastBySignal[iSignal  + "_" + str(iHorizon)] = lAutoF1
        lAutoF1.train(df , 'Date' , iSignal, iHorizon)
        self.reportModelInfo(lAutoF1);

    def computeModelPerfOnTraining(self, iModel):
        lPerfData = pd.DataFrame()
        self.mTrainPerfData[iSignal] = lPerfData;
        lPerfData = lPerfData.append(iModel.mSignalDecomposition.mTrPerfDetails)
        lPerfData.reset_index(inplace = True)
        #lPerfData.plot.line('level_0', 'ForecastMAPE')

    def trainSignal(self, iSignal, iHorizon):
        self.getTrainingDataset(iSignal, iHorizon);
        self.trainModel(iSignal, iHorizon);
        

    def testSignal(self, iSignal, iHorizon):
        start_time = time.time()
        self.trainSignal(iSignal, iHorizon);
        self.getTestPerfs(iSignal, iHorizon);
        end_time = time.time()
        self.mTrainTime[iSignal  + "_" + str(iHorizon)] = end_time - start_time;
        self.dumpForecastPerfs(iSignal, iHorizon);        

    def getApplyInDatset(self, iSignal, iHorizon):
        self.mApplyIn = pd.DataFrame();
        lSignalDataset = self.mTSSpec.mFullDataset;
        lFullDF = lSignalDataset[iSignal].dropna();
        #.astype(np.double)
        N = lFullDF.shape[0]
        lSize = N - iHorizon;
        self.checkHorizon(N , iHorizon);
        self.mApplyIn[iSignal] = lFullDF[0: lSize];
        self.mApplyIn['Date'] = range(0 , lSize);
        #self.mApplyIn.to_csv(iSignal + "_applyIn.csv");

    def applyModel(self, iSignal, iHorizon):
        lAutoF1 = self.mAutoForecastBySignal[iSignal  + "_" + str(iHorizon)]
        self.mApplyOut = lAutoF1.forecast(self.mApplyIn, iHorizon);
        #self.mApplyOut.to_csv(iSignal + "_applyOut.csv");
        # print(self.mApplyOut.tail());
        assert(self.mApplyOut.shape[0] == (iHorizon + self.mApplyIn.shape[0]));

    def reportActualAndPredictedData(self, iSignal, iHorizon):
        print("FORECAST_DETAIL_ACTUAL" , self.mTSSpec.mName , iSignal , self.mActual.values);
        print("FORECAST_DETAIL_PREDICTED" , self.mTSSpec.mName , iSignal , self.mPredicted.values);
        assert(self.mActual.shape[0] == iHorizon);
        assert(self.mPredicted.shape[0] == iHorizon);
        # print(self.mActual.describe());
        # print(self.mPredicted.describe());
        # print(self.mActual.tail());
        # print(self.mPredicted.tail());

    def computePerfOnForecasts(self, iSignal, iHorizon):
        lAutoF1 = self.mAutoForecastBySignal[iSignal  + "_" + str(iHorizon)]
        lForecastPerf = lAutoF1.computePerf(self.mActual, self.mPredicted, self.mBenchName + "_" + self.mTSSpec.mName + "_" + iSignal );
        self.mTestPerfData[iSignal  + "_" + str(iHorizon)] = lForecastPerf;


    def generateCode(self, iSignal, iHorizon):
        lAutoF = self.mAutoForecastBySignal[iSignal  + "_" + str(iHorizon)]
        lCodeGenerator = tscodegen.cTimeSeriesCodeGenerator();
        lSQL = lCodeGenerator.testGeneration(lAutoF);
        del lCodeGenerator;

    def getTestPerfs(self, iSignal, iHorizon):
        self.getApplyInDatset(iSignal, iHorizon);
        self.applyModel(iSignal, iHorizon);
        lSignalDataset = self.mTSSpec.mFullDataset;
        lFullDF = lSignalDataset[iSignal].dropna()
        self.mActual = lFullDF.tail(iHorizon);
        self.mPredicted = self.mApplyOut[iSignal + '_BestModelForecast'].tail(iHorizon);
        self.reportActualAndPredictedData(iSignal, iHorizon);
        self.computePerfOnForecasts(iSignal, iHorizon);
        self.generateCode(iSignal, iHorizon);
        
    def dumpForecastPerfs(self, iSignal, iHorizon):
        lAutoF1 = self.mAutoForecastBySignal[iSignal  + "_" + str(iHorizon)]
        lPerf = self.mTestPerfData[iSignal  + "_" + str(iHorizon)];
        print("BENCHMARK_PERF_DETAIL", self.mTSSpec.mName , iSignal,
              self.mTrainDataset[iSignal  + "_" + str(iHorizon)].shape[0] ,
              iHorizon,
              str(self.mTrainTime[iSignal  + "_" + str(iHorizon)]),
              lAutoF1.mSignalDecomposition.mBestTransformationName,
              lAutoF1.mSignalDecomposition.mBestTransformation.mBestModelName,
              lPerf.mCount,
              lPerf.mMAPE,  lPerf.mSMAPE);
        pass



class cGeneric_Tester:
    '''
    test a collection of signals
    '''

    def __init__(self , tsspec, bench_name):
        print("BENCH_DATA" , bench_name, tsspec)
        self.mTSSpec = tsspec;
        self.mBenchName = bench_name;
        self.mType = "OneDataFramePerSignal";
        if(hasattr(self.mTSSpec , "mFullDataset")):
            self.mType = "OneDataFrameForAllSignals";
        print("Bench type" , self.mType);
        self.fillSignalInfo();

    def fillSignalInfo(self):
        self.mTSSpecPerSignal = {};
        if(self.mType == "OneDataFrameForAllSignals"):
            lTSSpec = self.mTSSpec;
            for sig in self.mTSSpec.mFullDataset.columns:
                if(sig != "Date"):
                    self.mTSSpecPerSignal[sig] = self.mTSSpec;
        else:
            self.mTSSpecPerSignal = self.mTSSpec;

    def testAllSignals(self, iHorizon):
        for sig in self.mTSSpecPerSignal.keys():
            tester = cGeneric_OneSignal_Tester(self.mTSSpecPerSignal[sig] , self.mBenchName);
            tester.testSignal(sig, iHorizon);
            del tester;
        pass
    
    def testSignals(self, iSignals, iHorizon = 2):
        sigs = iSignals.split(" ");
        for sig in sigs:
            tester = cGeneric_OneSignal_Tester(self.mTSSpecPerSignal[sig] , self.mBenchName);
            tester.testSignal(sig, iHorizon);
            del tester;
        pass


    def run_multiprocessed(self, nbprocesses = 20):
        pool = mp.Pool(nbprocesses)
        args = []
        for sig in self.mTSSpecPerSignal.keys():
            a = cGeneric_Tester_Arg(self.mBenchName, self.mTSSpecPerSignal[sig], sig , 2);
            args = args + [a];
                
        asyncResult = pool.map_async(run_bench_process, args);
            
        resultList = asyncResult.get()


