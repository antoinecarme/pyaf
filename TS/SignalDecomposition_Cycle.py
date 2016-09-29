import pandas as pd
import numpy as np
import datetime
from . import SignalDecomposition_Time as tsti
from . import SignalDecomposition_Perf as tsperf
from . import SignalDecomposition_Quant as tsquant
from . import SignalDecomposition_Plots as tsplot

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
    

    def getCycleResidueName(self):
        return self.getCycleName() + "_residue";


    def plot(self):
        tsplot.decomp_plot(self.mCycleFrame, self.mTimeInfo.mNormalizedTimeColumn,
                           self.mTrend_residue_name, self.getCycleName() , self.getCycleResidueName());


    def computePerf(self):
        self.mCycleFitPerf = tsperf.cPerf();
        self.mCycleForecastPerf = tsperf.cPerf();
        (lFrameFit, lFrameForecast, lFrameTest) = self.mTimeInfo.cutFrame(self.mCycleFrame);
        
        self.mCycleFitPerf.compute(
            lFrameFit[self.mTrend_residue_name], lFrameFit[self.getCycleName()], self.getCycleName())
        self.mCycleForecastPerf.compute(
            lFrameForecast[self.mTrend_residue_name], lFrameForecast[self.getCycleName()], self.getCycleName())
    

class cZeroCycle(cAbstractCycle):

    def __init__(self , trend):
        super().__init__(trend);
        self.mFormula = "NoCycle"

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
        self.mCyclePerfs = pd.DataFrame()
        self.mCycleFrame = pd.DataFrame()
        self.mCyclePerfDict = {}
        self.mBestCycleValueDict = {}
        self.mBestCycleLength = 0
        self.mBestCyclePerf = {};
        self.mCriterion = criterion
        
    def getCycleName(self):
        return self.mTrend_residue_name + "_bestCycle_by" + self.mCriterion;

    def dumpCyclePerfs(self):
        print(self.mCyclePerfDict);

    def computeBestCycle(self):
        # self.dumpCyclePerfs();
        self.mBestCycleFrame = pd.DataFrame()
        self.mCycleFrame[self.getCycleName()] = np.zeros_like(self.mCycleFrame[self.mTrend_residue_name])
        self.mCycleFrame[self.getCycleResidueName()] = self.mCycleFrame[self.mTrend_residue_name]
        lCycleFrameEstim = self.mTimeInfo.getEstimPart(self.mCycleFrame);
        self.mBestCycleLength = 0;
        lBestCycleIdx = 0;
        lBestCriterion = None;
        if(self.mCyclePerfDict):
            for k in sorted(self.mCyclePerfDict.keys()):
                # smallest cycles are better
                if((lBestCriterion is None) or (self.mCyclePerfDict[k] < lBestCriterion)):
                    lBestCycleIdx = k;
                    lBestCriterion = self.mCyclePerfDict[k];
                    
            # lBestCycleIdx = min(self.mCyclePerfDict, key=self.mCyclePerfDict.get)
            
            best = self.mTrend_residue_name + '_Cycle_' + str(lBestCycleIdx)
            self.mBestCycleValueDict = lCycleFrameEstim.groupby([best])[self.mTrend_residue_name].mean().to_dict()
            self.mTimeInfo.addVars(self.mBestCycleFrame);
            self.mBestCycleFrame[self.mTrend_residue_name] = self.mCycleFrame[self.mTrend_residue_name]
            if(self.mCyclePerfDict[lBestCycleIdx] < self.mOptions.mCycle_Criterion_Threshold) :
                self.mBestCycleLength = lBestCycleIdx
                self.mBestCycleFrame[self.getCycleName()] = self.mCycleFrame[best + '_enc']
                self.mBestCycleFrame[self.getCycleResidueName()] =   self.mCycleFrame[self.mTrend_residue_name] - self.mCycleFrame[best + '_enc']
                self.mCycleFrame[self.getCycleName()] = self.mBestCycleFrame[self.getCycleName()];
                self.mCycleFrame[self.getCycleResidueName()] = self.mBestCycleFrame[self.getCycleResidueName()];
                self.mDefaultValue = lCycleFrameEstim[self.mTrend_residue_name].mean();
        # print("BEST_CYCLE_PERF" , self.mTrend_residue_name, self.mBestCycleLength)
        pass
        
    def generate_cycles(self):
        self.mTimeInfo.addVars(self.mCycleFrame);
        self.mCycleFrame[self.mTrend_residue_name ] = self.mTrendFrame[self.mTrend_residue_name]
        self.mCyclePerfs = {}
        self.mCyclePerfDict = {}
        lMaxRobustCycle = self.mTrendFrame.shape[0]/12;
        # print("MAX_ROBUST_CYCLE_LENGTH", self.mTrendFrame.shape[0], lMaxRobustCycle);
        for i in self.mOptions.mCycleLengths:
            if ((i > 1) and (i <= lMaxRobustCycle)):
                name_i = self.mTrend_residue_name + '_Cycle_' + str(i)
                self.mCycleFrame[name_i] = self.mCycleFrame[self.mTimeInfo.mRowNumberColumn] % i
                lCycleFrameEstim = self.mTimeInfo.getEstimPart(self.mCycleFrame);
                lTrendMeanEstim = lCycleFrameEstim[self.mTrend_residue_name].mean();
                lGroupBy = lCycleFrameEstim.groupby([name_i])[self.mTrend_residue_name].mean();
                lEncodedValueDict = lGroupBy.to_dict()
                lDefaultValue = lTrendMeanEstim;

                self.mCycleFrame[name_i + '_enc'] = self.mCycleFrame[name_i].apply(lambda x : lEncodedValueDict.get(x , lDefaultValue))

                lPerf = tsperf.cPerf();
                # validate the cycles on the last H values
                lValidFrame = self.mTimeInfo.getValidPart(self.mCycleFrame);
                lCritValue = lPerf.computeCriterion(lValidFrame[self.mTrend_residue_name] ,
                                                    lValidFrame[name_i + "_enc"],
                                                    self.mCriterion)
                self.mCyclePerfs[name_i] = [lPerf]
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
        if(self.mBestCycleLength > 0):
            self.mFormula = "Cycle_Length_" + str(self.mBestCycleLength);
            

    def transformDataset(self, df):
        df[self.getCycleName()] = np.zeros_like(df[self.mTimeInfo.mRowNumberColumn]);
        if(self.mBestCycleLength > 0):
            lValueCol = df[self.mTimeInfo.mRowNumberColumn].apply(lambda x : x % self.mBestCycleLength);
            df['cycle_internal'] = lValueCol;
            df[self.getCycleName()] = lValueCol.apply(lambda x : self.mBestCycleValueDict.get(x , self.mDefaultValue));

        target = df[self.mTrend_residue_name]
        df[self.getCycleResidueName()] = target - df[self.getCycleName()].values
        check_not_nan(self.mCycleFrame[self.getCycleName()].values , self.getCycleName());

        return df;

class cCycleEstimator:
    
    def __init__(self):
        self.mSignalFrame = pd.DataFrame()
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
                cycle.mSignalFrame = self.mSignalFrame;
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
                cycle.fit();
                cycle.computePerf();
                self.dumpCyclePerf(cycle)
                self.mCycleFrame[cycle.getCycleName()] = cycle.mCycleFrame[cycle.getCycleName()]
                self.mCycleFrame[cycle.getCycleResidueName()] = cycle.mCycleFrame[cycle.getCycleResidueName()]
                check_not_nan(self.mCycleFrame[cycle.getCycleResidueName()].values , cycle.getCycleResidueName())
        pass

    def estimateAllCycles(self):
        self.defineCycles();
        self.estimateCycles()
        
