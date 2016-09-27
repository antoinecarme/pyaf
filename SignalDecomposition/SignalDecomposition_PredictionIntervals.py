import pandas as pd
import numpy as np

from . import SignalDecomposition as sigdec

from . import SignalDecomposition_Perf as tsperf

class cPredictionIntervalsEstimator:
    
    def __init__(self):
        self.mSignalDecomposition = None;
        self.mSignalFrame = pd.DataFrame()
        self.mHorizon = -1;
        self.mFitPerformances = {}
        self.mForecastPerformances = {}
        self.mTestPerformances = {}

    def computePerformances(self):
        best_dec = self.mSignalDecomposition.mBestTransformation;
        lTimeColumn = self.mTime;
        lSignalColumn = self.mSignal;
        lForecastColumn = self.mSignal + "_BestModelForecast";
        df = self.mSignalFrame;
        N = df.shape[0];
        (lOriginalFit, lOriginalForecast, lOriginalTest) = best_dec.mTimeInfo.cutFrame(df);
        df1 = df.reset_index().copy();
        for h in range(0 , self.mHorizon):
            df2 = None;
            df2 = best_dec.forecastModelOneStepAhead(df1.copy());
            for exog in best_dec.mExogenousVariables:
                df2[ exog ] = df[ exog ];
            df2 = df2.head(N);
            # print(df1.info());
            # print(df2.info());
            # print(df1[[lTimeColumn, lSignalColumn]].head());
            # print(df2[[lTimeColumn, lSignalColumn, lForecastColumn]].head());
            # print(df1[[lTimeColumn, lSignalColumn]].tail());
            # print(df2[[lTimeColumn, lSignalColumn, lForecastColumn]].tail());
            lHorizonName = lForecastColumn + "_" + str(h + 1);
            (lFrameFit, lFrameForecast, lFrameTest) = best_dec.mTimeInfo.cutFrame(df2);
            self.mFitPerformances[lHorizonName] = tsperf.cPerf();
            self.mFitPerformances[lHorizonName].compute(lOriginalFit[lSignalColumn], lFrameFit[lForecastColumn], lHorizonName);
            self.mForecastPerformances[lHorizonName] = tsperf.cPerf();
            self.mForecastPerformances[lHorizonName].compute(lOriginalForecast[lSignalColumn], lFrameForecast[lForecastColumn], lHorizonName);
            self.mTestPerformances[lHorizonName] = tsperf.cPerf();
            self.mTestPerformances[lHorizonName].compute(lOriginalTest[lSignalColumn], lFrameTest[lForecastColumn], lHorizonName);
            df1[lSignalColumn] = df2[lForecastColumn];
        # self.dump();

    def dump_detailed(self):
        lForecastColumn = self.mSignal + "_BestModelForecast";
        for h in range(0 , self.mHorizon):
            lHorizonName = lForecastColumn + "_" + str(h + 1);
            hn = lHorizonName;
            print("CONFIDENCE_INTERVAL_DUMP_FIT" , hn , self.mFitPerformances[hn].mL2 ,  self.mFitPerformances[hn].mMAPE);
            print("CONFIDENCE_INTERVAL_DUMP_FORECAST" , hn , self.mForecastPerformances[hn].mL2 ,  self.mForecastPerformances[hn].mMAPE);
            print("CONFIDENCE_INTERVAL_DUMP_TEST" , hn , self.mTestPerformances[hn].mL2 ,  self.mTestPerformances[hn].mMAPE);


    def dump(self):
        lForecastColumn = self.mSignal + "_BestModelForecast";
        for h in range(0 , self.mHorizon):
            lHorizonName = lForecastColumn + "_" + str(h + 1);
            hn = lHorizonName;
            print("CONFIDENCE_INTERVAL_DUMP_FORECAST" , hn , self.mForecastPerformances[hn].mL2);
            
