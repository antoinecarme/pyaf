import pandas as pd
import numpy as np
import datetime
from . import Time as tsti
from . import Perf as tsperf
from . import Plots as tsplot

# for timing
import time

def check_not_nan(sig, name):
    #    print("check_not_nan "  + name);
    #    print(sig);
    if(np.isnan(sig).any()):
        raise ValueError("Invalid cycle '" + name + "'");
    pass


class cAbstractCycle:
    def __init__(self , trend):
        self.mTimeInfo = tsti.cTimeInfo()
        self.mTrendFrame = pd.DataFrame()
        self.mCycleFrame = pd.DataFrame()
        self.mTrend = trend;
        self.mTrend_residue_name = self.mTrend.mOutName + '_residue'
        self.mFormula = None;
        self.mComplexity = None;
    

    def getCycleResidueName(self):
        return self.getCycleName() + "_residue";


    def plot(self):
        tsplot.decomp_plot(self.mCycleFrame, self.mTimeInfo.mNormalizedTimeColumn,
                           self.mTrend_residue_name, self.getCycleName() , self.getCycleResidueName());


    def computePerf(self):
        self.mCycleFitPerf = tsperf.cPerf();
        self.mCycleForecastPerf = tsperf.cPerf();
        # self.mCycleFrame[[self.mTrend_residue_name, self.getCycleName()]].to_csv(self.getCycleName() + ".csv");
        (lFrameFit, lFrameForecast, lFrameTest) = self.mTimeInfo.cutFrame(self.mCycleFrame);
        
        self.mCycleFitPerf.compute(
            lFrameFit[self.mTrend_residue_name], lFrameFit[self.getCycleName()], self.getCycleName())
        self.mCycleForecastPerf.compute(
            lFrameForecast[self.mTrend_residue_name], lFrameForecast[self.getCycleName()], self.getCycleName())
    

class cZeroCycle(cAbstractCycle):

    def __init__(self , trend):
        super().__init__(trend);
        self.mFormula = "NoCycle"
        self.mComplexity = 0;

    def getCycleName(self):
        return self.mTrend_residue_name + "_zeroCycle";


    def fit(self):
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mTimeInfo.addVars(self.mCycleFrame);
        self.mCycleFrame[self.mTrend_residue_name] = self.mTrendFrame[self.mTrend_residue_name]
        self.mCycleFrame[self.getCycleName()] = np.zeros_like(self.mTrendFrame[self.mTrend_residue_name])
        self.mCycleFrame[self.getCycleResidueName()] = self.mCycleFrame[self.mTrend_residue_name];
        self.mOutName = self.getCycleName()
        
    def transformDataset(self, df):
        target = df[self.mTrend_residue_name]
        df[self.getCycleName()] = np.zeros_like(df[self.mTrend_residue_name]);
        df[self.getCycleResidueName()] = target - df[self.getCycleName()].values        
        return df;

class cSeasonalPeriodic(cAbstractCycle):
    def __init__(self , trend, date_part):
        super().__init__(trend);
        self.mDatePart = date_part;
        self.mEncodedValueDict = {}
        self.mFormula = "Seasonal_" + self.mDatePart;
        self.mComplexity = 1;
        
    def getCycleName(self):
        return self.mTrend_residue_name + "_Seasonal_" + self.mDatePart;



    def fit(self):
        assert(self.mTimeInfo.isPhysicalTime());
        lHor = self.mTimeInfo.mHorizon;
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mTimeInfo.addVars(self.mCycleFrame);
        lName = self.getCycleName();
        self.mCycleFrame[self.mTrend_residue_name] = self.mTrendFrame[self.mTrend_residue_name]
        self.mCycleFrame[lName] = self.mTrendFrame[self.mTime].apply(
            lambda x : self.mTimeInfo.get_date_part_value(x , self.mDatePart));
        # we encode only using estimation
        lCycleFrameEstim = self.mTimeInfo.getEstimPart(self.mCycleFrame);
        lTrendMeanEstim = lCycleFrameEstim[self.mTrend_residue_name].mean();
        lGroupBy = lCycleFrameEstim.groupby([lName])[self.mTrend_residue_name].mean(); 
        self.mEncodedValueDict = lGroupBy.to_dict()
        self.mDefaultValue = lTrendMeanEstim;
        # print("cSeasonalPeriodic_DefaultValue" , self.getCycleName(), self.mDefaultValue);

        self.mCycleFrame[lName + '_enc'] = self.mCycleFrame[lName].apply(lambda x : self.mEncodedValueDict.get(x , self.mDefaultValue))
        self.mCycleFrame[lName + '_enc'].fillna(lTrendMeanEstim, inplace=True);
        self.mCycleFrame[self.getCycleResidueName()] = self.mCycleFrame[self.mTrend_residue_name] - self.mCycleFrame[lName + '_enc'];
        self.mCycleFrame[lName + '_NotEncoded'] = self.mCycleFrame[lName];
        self.mCycleFrame[lName] = self.mCycleFrame[lName + '_enc'];
        
        self.mOutName = self.getCycleName()
        #print("encoding '" + lName + "' " + str(self.mEncodedValueDict));
        

    def transformDataset(self, df):
        target = df[self.mTrend_residue_name]
        lDatePartValues = df[self.mTime].apply(
            lambda x : self.mTimeInfo.get_date_part_value(x , self.mDatePart));
        df[self.getCycleName()] = lDatePartValues.apply(lambda x : self.mEncodedValueDict.get(x , self.mDefaultValue));
        df[self.getCycleResidueName()] = target - df[self.getCycleName()].values        
        return df;

class cBestCycleForTrend(cAbstractCycle):
    def __init__(self , trend, criterion):
        super().__init__(trend);
        self.mCycleFrame = pd.DataFrame()
        self.mCyclePerfDict = {}
        self.mBestCycleValueDict = {}
        self.mBestCycleLength = None
        self.mCriterion = criterion
        self.mComplexity = 2;
        
    def getCycleName(self):
        return self.mTrend_residue_name + "_bestCycle_by" + self.mCriterion;

    def dumpCyclePerfs(self):
        print(self.mCyclePerfDict);

    def computeBestCycle(self):
        # self.dumpCyclePerfs();
        lCycleFrameEstim = self.mTimeInfo.getEstimPart(self.mCycleFrame);
        self.mDefaultValue = lCycleFrameEstim[self.mTrend_residue_name].mean();
        self.mBestCycleLength = None;
        lBestCycleIdx = None;
        lBestCriterion = None;
        if(self.mCyclePerfDict):
            for k in sorted(self.mCyclePerfDict.keys()):
                # smallest cycles are better
                if((lBestCriterion is None) or (self.mCyclePerfDict[k] < lBestCriterion)):
                    lBestCycleIdx = k;
                    lBestCriterion = self.mCyclePerfDict[k];
                    
            if(self.mOptions.mCycle_Criterion_Threshold is None or                 
                (self.mCyclePerfDict[lBestCycleIdx] < self.mOptions.mCycle_Criterion_Threshold)) :
                self.mBestCycleLength = lBestCycleIdx
        # print("BEST_CYCLE_PERF" , self.mTrend_residue_name, self.mBestCycleLength)


        self.transformDataset(self.mCycleFrame);
        pass


    def generate_cycles(self):
        self.mTimeInfo.addVars(self.mCycleFrame);
        self.mCycleFrame[self.mTrend_residue_name ] = self.mTrendFrame[self.mTrend_residue_name]
        lCycleFrameEstim = self.mTimeInfo.getEstimPart(self.mCycleFrame);
        self.mDefaultValue = lCycleFrameEstim[self.mTrend_residue_name].mean();
        del lCycleFrameEstim;
        self.mCyclePerfDict = {}
        lMaxRobustCycle = self.mTrendFrame.shape[0]/12;
        # print("MAX_ROBUST_CYCLE_LENGTH", self.mTrendFrame.shape[0], lMaxRobustCycle);
        
        lCycleFrame = pd.DataFrame();
        lCycleFrame[self.mTrend_residue_name ] = self.mTrendFrame[self.mTrend_residue_name]
        for i in self.mOptions.mCycleLengths:
            if ((i > 1) and (i <= lMaxRobustCycle)):
                name_i = self.mTrend_residue_name + '_Cycle';
                lCycleFrame[name_i] = self.mCycleFrame[self.mTimeInfo.mRowNumberColumn] % i
                lCycleFrameEstim = self.mTimeInfo.getEstimPart(lCycleFrame);
                lGroupBy = lCycleFrameEstim.groupby([name_i])[self.mTrend_residue_name].mean();
                lEncodedValueDict = lGroupBy.to_dict()
                lCycleFrame[name_i + '_enc'] = lCycleFrame[name_i].apply(
                    lambda x : lEncodedValueDict.get(x , self.mDefaultValue))

                self.mBestCycleValueDict[i] = lEncodedValueDict;
                
                lPerf = tsperf.cPerf();
                # validate the cycles on the last H values
                lValidFrame = self.mTimeInfo.getValidPart(lCycleFrame);
                lCritValue = lPerf.computeCriterion(lValidFrame[self.mTrend_residue_name],
                                                    lValidFrame[name_i + "_enc"],
                                                    self.mCriterion)
                self.mCyclePerfDict[i] = lPerf.getCriterionValue(lCritValue);
        pass

    def fit(self):
        # print("cycle_fit" , self.mTrend_residue_name);
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.generate_cycles();
        self.computeBestCycle();
        self.mOutName = self.getCycleName()
        self.mFormula = "Cycle_None"
        if(self.mBestCycleLength is not None):
            self.mFormula = "Cycle_Length_" + str(self.mBestCycleLength);
        self.transformDataset(self.mCycleFrame);

    def transformDataset(self, df):
        if(self.mBestCycleLength is not None):
            lValueCol = df[self.mTimeInfo.mRowNumberColumn].apply(lambda x : x % self.mBestCycleLength);
            df['cycle_internal'] = lValueCol;
            lDict = self.mBestCycleValueDict[self.mBestCycleLength];
            df[self.getCycleName()] = lValueCol.apply(lambda x : lDict.get(x , self.mDefaultValue));
        else:
            df[self.getCycleName()] = np.zeros_like(df[self.mTimeInfo.mRowNumberColumn]);            

        target = df[self.mTrend_residue_name]
        df[self.getCycleResidueName()] = target - df[self.getCycleName()].values
        check_not_nan(self.mCycleFrame[self.getCycleName()].values , self.getCycleName());

        return df;

class cCycleEstimator:
    
    def __init__(self):
        self.mTimeInfo = tsti.cTimeInfo()
        self.mTrendFrame = pd.DataFrame()
        self.mCycleFrame = pd.DataFrame()
        self.mCycleList = {}

    def addSeasonal(self, trend, seas_type, resolution):
        if(resolution >= self.mTimeInfo.mResolution):
            self.mCycleList[trend] = self.mCycleList[trend] + [cSeasonalPeriodic(trend, seas_type)];
        pass
    
    def defineCycles(self):
        for trend in self.mTrendList:
            self.mCycleList[trend] = [cZeroCycle(trend)];
            if(self.mOptions.mEnableCycles):
                self.mCycleList[trend] = self.mCycleList[trend] + [
                    cBestCycleForTrend(trend, self.mOptions.mCycle_Criterion)];
            if(self.mOptions.mEnableSeasonals and self.mTimeInfo.isPhysicalTime()):
                self.addSeasonal(trend, "MonthOfYear", self.mTimeInfo.RES_MONTH);
                self.addSeasonal(trend, "DayOfMonth", self.mTimeInfo.RES_DAY);
                self.addSeasonal(trend, "Hour", self.mTimeInfo.RES_HOUR);
                self.addSeasonal(trend, "Minute", self.mTimeInfo.RES_MINUTE);
                self.addSeasonal(trend, "Second", self.mTimeInfo.RES_SECOND);

                self.addSeasonal(trend, "WeekOfYear", self.mTimeInfo.RES_DAY);
                self.addSeasonal(trend, "DayOfWeek", self.mTimeInfo.RES_DAY);
                
        for trend in self.mTrendList:
            for cycle in self.mCycleList[trend]:
                cycle.mTrendFrame = self.mTrendFrame;
                cycle.mTimeInfo = self.mTimeInfo;
                cycle.mOptions = self.mOptions;
            
    def plotCycles(self):
        for trend in self.mTrendList:
            for cycle in self.mCycleList[trend]:
                cycle.plot()

    

    def dumpCyclePerf(self, cycle):
        if(self.mOptions.mDebugCycles):
            print("CYCLE_PERF_DETAIL_COUNT_FIT_FORECAST" , cycle.mOutName,
                  "%.3f" % (cycle.mCycleFitPerf.mCount), "%.3f" % (cycle.mCycleForecastPerf.mCount));
            print("CYCLE_PERF_DETAIL_MAPE_FIT_FORECAST" , cycle.mOutName,
                  "%.3f" % (cycle.mCycleFitPerf.mMAPE), "%.3f" % (cycle.mCycleForecastPerf.mMAPE));
            print("CYCLE_PERF_DETAIL_L2_FIT_FORECAST" , cycle.mOutName,
                  "%.3f" % (cycle.mCycleFitPerf.mL2),  "%.3f" % (cycle.mCycleForecastPerf.mL2));
            print("CYCLE_PERF_DETAIL_R2_FIT_FORECAST" , cycle.mOutName,
                  "%.3f" % (cycle.mCycleFitPerf.mR2),  "%.3f" % (cycle.mCycleForecastPerf.mR2));
            print("CYCLE_PERF_DETAIL_PEARSONR_FIT_FORECAST" , cycle.mOutName,
                  "%.3f" % (cycle.mCycleFitPerf.mPearsonR),  "%.3f" % (cycle.mCycleForecastPerf.mPearsonR));


    def estimateCycles(self):
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mTimeInfo.addVars(self.mCycleFrame);
        for trend in self.mTrendList:
            lTrend_residue_name = trend.mOutName + '_residue'
            self.mCycleFrame[lTrend_residue_name] = self.mTrendFrame[lTrend_residue_name]
            for cycle in self.mCycleList[trend]:
                start_time = time.time()
                cycle.fit();
                cycle.computePerf();
                self.dumpCyclePerf(cycle)
                self.mCycleFrame[cycle.getCycleName()] = cycle.mCycleFrame[cycle.getCycleName()]
                self.mCycleFrame[cycle.getCycleResidueName()] = cycle.mCycleFrame[cycle.getCycleResidueName()]
                check_not_nan(self.mCycleFrame[cycle.getCycleResidueName()].values , cycle.getCycleResidueName())
                end_time = time.time()
                lTrainingTime = round(end_time - start_time , 2);
                if(self.mOptions.mDebugProfile):
                    print("CYCLE_TRAINING_TIME_IN_SECONDS '" + cycle.mOutName + "' " + str(lTrainingTime))
        pass

    def estimateAllCycles(self):
        self.defineCycles();
        self.estimateCycles()
        
