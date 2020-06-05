# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np

from . import Time as tsti
from . import DateTime_Functions as dtfunc
from . import Perf as tsperf
from . import Plots as tsplot
from . import Utils as tsutil

# for timing
import time


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
                           self.mTrend_residue_name, self.getCycleName() , self.getCycleResidueName(), horizon = self.mTimeInfo.mHorizon);


    def check_not_nan(self, sig , name):
        #print("check_not_nan");
        if(np.isnan(sig).any() or np.isinf(sig).any() ):
            logger = tsutil.get_pyaf_logger();
            logger.error("CYCLE_RESIDUE_WITH_NAN_IN_SIGNAL" + str(sig));
            raise tsutil.Internal_PyAF_Error("CYCLE_COLUMN _FOR_TREND_RESIDUE ['"  + name + "'");
        pass


    def compute_target_means_by_cycle_value(self , iCycleFrame, iCycleName):
        # we encode only using estimation
        lCycleFrameEstim = self.mSplit.getEstimPart(iCycleFrame);
        lGroupBy = lCycleFrameEstim.groupby(by=[iCycleName] , sort=False)[self.mTrend_residue_name]
        lEncodedValueDict = None
        if(self.mOptions.mCycle_Encoding_Scheme == "Target_Mean"):
            lEncodedValueDict = lGroupBy.mean().to_dict();
        else:
            lEncodedValueDict = lGroupBy.median().to_dict();
        return lEncodedValueDict

    def compute_target_means_default_value(self):
        # we encode only using estimation
        lCycleFrameEstim = self.mSplit.getEstimPart(self.mCycleFrame);
        if(self.mOptions.mCycle_Encoding_Scheme == "Target_Mean"):
            return lCycleFrameEstim[self.mTrend_residue_name].mean();
        return lCycleFrameEstim[self.mTrend_residue_name].median();

    def computePerf(self):
        if(self.mOptions.mDebug):
            self.check_not_nan(self.mCycleFrame[self.getCycleResidueName()], self.getCycleResidueName())
        # self.mCycleFrame.to_csv(self.getCycleResidueName() + ".csv");
        self.mCycleFitPerf = tsperf.cPerf();
        self.mCycleForecastPerf = tsperf.cPerf();
        # self.mCycleFrame[[self.mTrend_residue_name, self.getCycleName()]].to_csv(self.getCycleName() + ".csv");
        (lFrameFit, lFrameForecast, lFrameTest) = self.mSplit.cutFrame(self.mCycleFrame);
        
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


    def dump_values(self):
        logger = tsutil.get_pyaf_logger();
        lDict = {}
        logger.info("ZERO_CYCLE_MODEL_VALUES " + self.getCycleName() + " 0.0 " + "{}");        
    
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
        self.mFormula = "Seasonal_" + self.mDatePart.name;
        self.mComplexity = 1;
        
        
    def getCycleName(self):
        return self.mTrend_residue_name + "_Seasonal_" + self.mDatePart.name;

    def dump_values(self):
        logger = tsutil.get_pyaf_logger();
        lDict = {}
        logger.info("SEASONAL_MODEL_VALUES " + self.getCycleName() + " " + str(self.mDefaultValue) + " " + str(self.mEncodedValueDict));


    def hasEnoughData(self, iTimeMin, iTimeMax):
        lTimeDelta = iTimeMax - iTimeMin;
        lDays = lTimeDelta / np.timedelta64(1,'D');
        lSeconds = lTimeDelta / np.timedelta64(1,'s');
        # these are just guessses of how much dataa is needed to get valid signal stats/means of each seasonal unit.
        # TODO : add these in the options. (None, None) => no limit
        lThresholds = {
            dtfunc.eDatePart.Hour : (1 * 10 , None), # 10 days
            dtfunc.eDatePart.Minute : (None , 3600 * 10), # 10 hours
            dtfunc.eDatePart.Second : (None , 360 * 10), # 10 minutes
            dtfunc.eDatePart.DayOfMonth : (30 * 10 , None), # 10 months
            dtfunc.eDatePart.DayOfWeek : (7 * 10 , None), # 10 weeks
            dtfunc.eDatePart.MonthOfYear : (360 * 10 , None), # 10 years
            dtfunc.eDatePart.WeekOfYear : (360 * 10 , None), # 10 years
            dtfunc.eDatePart.WeekOfYear : (7 * 10 , None), # 10 weeks
            dtfunc.eDatePart.DayOfYear : (360 * 10 , None), # 10 years
            dtfunc.eDatePart.HourOfWeek : (7 * 10 , None), # 10 weeks
            dtfunc.eDatePart.TwoHourOfWeek : (7 * 10 , None), # 10 weeks
            dtfunc.eDatePart.ThreeHourOfWeek : (7 * 10 , None), # 10 weeks
            dtfunc.eDatePart.FourHourOfWeek : (7 * 10 , None), # 10 weeks
            dtfunc.eDatePart.SixHourOfWeek : (7 * 10 , None), # 10 weeks
            dtfunc.eDatePart.EightHourOfWeek : (7 * 10 , None), # 10 weeks
            dtfunc.eDatePart.TwelveHourOfWeek : (7 * 10 , None), # 10 weeks
            dtfunc.eDatePart.WeekOfMonth : (30 * 10 , None), # 10 months
            dtfunc.eDatePart.DayOfNthWeekOfMonth : (30 * 10 , None) # 10 months
            }

        lThreshold = lThresholds.get(self.mDatePart)
        if(lThreshold[0] is not None):
            return (lDays >= lThreshold[0]);
        elif(lThreshold[1] is not None):
            return (lSeconds >= lThreshold[1]);        
        return False;




    def compute_date_parts(self, iTimeValues):
        lHelper = dtfunc.cDateTime_Helper()
        return lHelper.apply_date_time_computer(self.mDatePart, iTimeValues);

        
    def fit(self):
        assert(self.mTimeInfo.isPhysicalTime());
        lHor = self.mTimeInfo.mHorizon;
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mTimeInfo.addVars(self.mCycleFrame);
        lName = self.getCycleName();
        self.mCycleFrame[self.mTrend_residue_name] = self.mTrendFrame[self.mTrend_residue_name]
        self.mCycleFrame[lName] = self.compute_date_parts(self.mTrendFrame[self.mTime])
        self.mDefaultValue = self.compute_target_means_default_value()
        self.mEncodedValueDict = self.compute_target_means_by_cycle_value(self.mCycleFrame, self.getCycleName())

        self.mCycleFrame[lName + '_enc'] = self.mCycleFrame[lName].apply(lambda x : self.mEncodedValueDict.get(x , self.mDefaultValue))
        self.mCycleFrame[lName + '_enc'].fillna(self.mDefaultValue, inplace=True);
        self.mCycleFrame[self.getCycleResidueName()] = self.mCycleFrame[self.mTrend_residue_name] - self.mCycleFrame[lName + '_enc'];
        self.mCycleFrame[lName + '_NotEncoded'] = self.mCycleFrame[lName];
        self.mCycleFrame[lName] = self.mCycleFrame[lName + '_enc'];
        
        self.mOutName = self.getCycleName()
        #print("encoding '" + lName + "' " + str(self.mEncodedValueDict));

    def transformDataset(self, df):
        target = df[self.mTrend_residue_name]
        lDateParts = self.compute_date_parts(df[self.mTime])
        df[self.getCycleName()] = lDateParts.apply(lambda x : self.mEncodedValueDict.get(x , self.mDefaultValue))
        df[self.getCycleResidueName()] = target - df[self.getCycleName()].values        
        return df;

class cBestCycleForTrend(cAbstractCycle):
    def __init__(self , trend, criterion):
        super().__init__(trend);
        self.mCycleFrame = pd.DataFrame()
        self.mCyclePerfByLength = {}
        self.mBestCycleValueDict = {}
        self.mBestCycleLength = None
        self.mCriterion = criterion
        self.mComplexity = 2;
        self.mFormula = "BestCycle"
        
    def getCycleName(self):
        return self.mTrend_residue_name + "_bestCycle_by" + self.mCriterion;

    def dump_values(self):
        logger = tsutil.get_pyaf_logger();
        lDict = {} if(self.mBestCycleLength is None) else self.mBestCycleValueDict[self.mBestCycleLength]
        logger.info("BEST_CYCLE_LENGTH_VALUES " + self.getCycleName() + " " + str(self.mBestCycleLength) + " " + str(self.mDefaultValue) + " " + str(lDict));

    
    def dumpCyclePerfs(self):
        print(self.mCyclePerfByLength);

    def computeBestCycle(self):
        # self.dumpCyclePerfs();
        self.mBestCycleLength = None;
        lData = self.mCyclePerfByLength.items()
        if(len(lData) == 0):
            return
        
        lPerf = tsperf.cPerf();
        # less MAPE is better, less categories is better, the last is the length to have a total order.
        lSortingMethod_By_MAPE = lambda x : (x[1][0], x[0])
        lData = sorted(lData, key = lSortingMethod_By_MAPE)
        assert(len(lData) > 0)
        lBestCriterion = lData[0][1]
        lData_smallest = [x for x in lData if lPerf.is_close_criterion_value(self.mOptions.mCycle_Criterion,
                                                                             x[1][0],
                                                                             iTolerance = 0.05, iRefValue = lBestCriterion[0])]
        lSortingMethod_By_Complexity = lambda x : (x[1][1], x[0])
        lData_smallest = sorted(lData_smallest, key = lSortingMethod_By_Complexity)
        assert(len(lData_smallest) > 0)
        self.mBestCycleLength = lData_smallest[0][0]

        self.transformDataset(self.mCycleFrame);
        pass


    def generate_cycles(self):
        self.mTimeInfo.addVars(self.mCycleFrame);
        self.mCycleFrame[self.mTrend_residue_name ] = self.mTrendFrame[self.mTrend_residue_name]
        self.mDefaultValue = self.compute_target_means_default_value();
        self.mCyclePerfByLength = {}
        lMaxRobustCycleLength = self.mTrendFrame.shape[0]//12;
        # print("MAX_ROBUST_CYCLE_LENGTH", self.mTrendFrame.shape[0], lMaxRobustCycleLength);
        lCycleLengths = self.mOptions.mCycleLengths or range(2,lMaxRobustCycleLength + 1)
        lCycleFrame = pd.DataFrame();
        lCycleFrame[self.mTrend_residue_name ] = self.mTrendFrame[self.mTrend_residue_name]
        for lLength in lCycleLengths:
            if ((lLength > 1) and (lLength <= lMaxRobustCycleLength)):
                name_length = self.mTrend_residue_name + '_Cycle';
                lCycleFrame[name_length] = self.mCycleFrame[self.mTimeInfo.mRowNumberColumn] % lLength
                lEncodedValueDict = self.compute_target_means_by_cycle_value(lCycleFrame, name_length)
                lCycleFrame[name_length + '_enc'] = lCycleFrame[name_length].apply(
                    lambda x : lEncodedValueDict.get(x , self.mDefaultValue))

                self.mBestCycleValueDict[lLength] = lEncodedValueDict;
                
                lPerf = tsperf.cPerf();
                # validate the cycles on the validation part
                lValidFrame = self.mSplit.getValidPart(lCycleFrame);
                lCritValue = lPerf.computeCriterion(lValidFrame[self.mTrend_residue_name],
                                                    lValidFrame[name_length + "_enc"],
                                                    self.mCriterion,
                                                    "Validation")
                if(lPerf.is_acceptable_criterion_value(self.mOptions.mCycle_Criterion, iRefValue = lCritValue)):
                    self.mCyclePerfByLength[lLength] = (round(lCritValue, 3) , len(lEncodedValueDict))
                    if(self.mOptions.mDebugCycles):
                        logger = tsutil.get_pyaf_logger();
                        logger.debug("CYCLE_INTERNAL_CRITERION " + name_length + " " + str(lLength) + \
                                     " " + self.mCriterion +" " + str(lCritValue))
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
            self.mFormula = "Cycle" #  + str(self.mBestCycleLength);
        self.transformDataset(self.mCycleFrame);

    def transformDataset(self, df):
        if(self.mBestCycleLength is not None):
            lValueCol = df[self.mTimeInfo.mRowNumberColumn].apply(lambda x : x % self.mBestCycleLength);
            df['cycle_internal'] = lValueCol;
            # print("BEST_CYCLE" , self.mBestCycleLength)
            # print(self.mBestCycleValueDict);
            lDict = self.mBestCycleValueDict[self.mBestCycleLength];
            df[self.getCycleName()] = lValueCol.apply(lambda x : lDict.get(x , self.mDefaultValue));
        else:
            df[self.getCycleName()] = np.zeros_like(df[self.mTimeInfo.mRowNumberColumn]);            

        target = df[self.mTrend_residue_name]
        df[self.getCycleResidueName()] = target - df[self.getCycleName()].values
        if(self.mOptions.mDebug):
            self.check_not_nan(self.mCycleFrame[self.getCycleName()].values , self.getCycleName());

        return df;

class cCycleEstimator:
    
    def __init__(self):
        self.mTimeInfo = tsti.cTimeInfo()
        self.mTrendFrame = pd.DataFrame()
        self.mCycleFrame = pd.DataFrame()
        self.mCycleList = {}
        
    def addSeasonal(self, trend, seas_type, resolution):
        if(resolution >= self.mTimeInfo.mResolution):
            lSeasonal = cSeasonalPeriodic(trend, seas_type);
            if(self.mOptions.mActivePeriodics[lSeasonal.mFormula]):
                if(lSeasonal.hasEnoughData(self.mTimeInfo.mTimeMin, self.mTimeInfo.mTimeMax)):
                    self.mCycleList[trend] = self.mCycleList[trend] + [lSeasonal];
                else:
                    if(self.mOptions.mDebugCycles):
                        lTimeDelta = self.mTimeInfo.mTimeMax - self.mTimeInfo.mTimeMin
                        lDays = lTimeDelta / np.timedelta64(1,'D');
                        logger = tsutil.get_pyaf_logger();
                        logger.debug("NOT_ENOUGH_DATA_TO_ANAYLSE_SEASONAL_PATTERN " + str((trend.__class__.__name__, seas_type, resolution, lDays)))

        pass
    
    def defineCycles(self):
        for trend in self.mTrendList:
            self.mCycleList[trend] = [];

            if(self.mOptions.mActivePeriodics['NoCycle']):
                self.mCycleList[trend] = [cZeroCycle(trend)];
            if(self.mOptions.mActivePeriodics['BestCycle']):
                self.mCycleList[trend] = self.mCycleList[trend] + [
                    cBestCycleForTrend(trend, self.mOptions.mCycle_Criterion)];
            if(self.mTimeInfo.isPhysicalTime()):
                # The order used here is mandatory. see filterSeasonals before changing this order.
                self.addSeasonal(trend, dtfunc.eDatePart.MonthOfYear, dtfunc.eTimeResolution.MONTH);
                self.addSeasonal(trend, dtfunc.eDatePart.WeekOfYear, dtfunc.eTimeResolution.DAY);
                self.addSeasonal(trend, dtfunc.eDatePart.DayOfMonth, dtfunc.eTimeResolution.DAY);
                self.addSeasonal(trend, dtfunc.eDatePart.DayOfWeek, dtfunc.eTimeResolution.DAY);
                self.addSeasonal(trend, dtfunc.eDatePart.DayOfYear, dtfunc.eTimeResolution.DAY);
                self.addSeasonal(trend, dtfunc.eDatePart.Hour, dtfunc.eTimeResolution.HOUR);
                self.addSeasonal(trend, dtfunc.eDatePart.Minute, dtfunc.eTimeResolution.MINUTE);
                self.addSeasonal(trend, dtfunc.eDatePart.Second, dtfunc.eTimeResolution.SECOND);
                self.addSeasonal(trend, dtfunc.eDatePart.HourOfWeek, dtfunc.eTimeResolution.HOUR);
                self.addSeasonal(trend, dtfunc.eDatePart.TwoHourOfWeek, dtfunc.eTimeResolution.HOUR);
                self.addSeasonal(trend, dtfunc.eDatePart.ThreeHourOfWeek, dtfunc.eTimeResolution.HOUR);
                self.addSeasonal(trend, dtfunc.eDatePart.FourHourOfWeek, dtfunc.eTimeResolution.HOUR);
                self.addSeasonal(trend, dtfunc.eDatePart.SixHourOfWeek, dtfunc.eTimeResolution.HOUR);
                self.addSeasonal(trend, dtfunc.eDatePart.EightHourOfWeek, dtfunc.eTimeResolution.HOUR);
                self.addSeasonal(trend, dtfunc.eDatePart.TwelveHourOfWeek, dtfunc.eTimeResolution.HOUR);
                self.addSeasonal(trend, dtfunc.eDatePart.WeekOfMonth, dtfunc.eTimeResolution.DAY);
                self.addSeasonal(trend, dtfunc.eDatePart.DayOfNthWeekOfMonth, dtfunc.eTimeResolution.DAY);
                

                
        for trend in self.mTrendList:
            if(len(self.mCycleList[trend]) == 0):
                self.mCycleList[trend] = [cZeroCycle(trend)];
            for cycle in self.mCycleList[trend]:
                cycle.mTrendFrame = self.mTrendFrame;
                cycle.mTimeInfo = self.mTimeInfo;
                cycle.mSplit = self.mSplit;
                cycle.mOptions = self.mOptions;
            
    def plotCycles(self):
        for trend in self.mTrendList:
            for cycle in self.mCycleList[trend]:
                cycle.plot()

    

    def dumpCyclePerf(self, cycle):
        if(self.mOptions.mDebugCycles):
            logger = tsutil.get_pyaf_logger();
            logger.debug("CYCLE_PERF_DETAIL_COUNT_FIT_FORECAST "  + cycle.mOutName +
                  " %.3f" % (cycle.mCycleFitPerf.mCount) + " %.3f" % (cycle.mCycleForecastPerf.mCount));
            logger.debug("CYCLE_PERF_DETAIL_MAPE_FIT_FORECAST " + cycle.mOutName +
                  " %.3f" % (cycle.mCycleFitPerf.mMAPE)+ " %.3f" % (cycle.mCycleForecastPerf.mMAPE));
            logger.debug("CYCLE_PERF_DETAIL_L2_FIT_FORECAST " + cycle.mOutName +
                  " %.3f" % (cycle.mCycleFitPerf.mL2) +  " %.3f" % (cycle.mCycleForecastPerf.mL2));
            logger.debug("CYCLE_PERF_DETAIL_R2_FIT_FORECAST " + cycle.mOutName +
                  " %.3f" % (cycle.mCycleFitPerf.mR2) +  " %.3f" % (cycle.mCycleForecastPerf.mR2));
            logger.debug("CYCLE_PERF_DETAIL_PEARSONR_FIT_FORECAST " + cycle.mOutName +
                  " %.3f" % (cycle.mCycleFitPerf.mPearsonR) +  " %.3f" % (cycle.mCycleForecastPerf.mPearsonR));


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
                if(self.mOptions.mDebugPerformance):
                    cycle.computePerf();
                self.dumpCyclePerf(cycle)
                self.mCycleFrame[cycle.getCycleName()] = cycle.mCycleFrame[cycle.getCycleName()]
                self.mCycleFrame[cycle.getCycleResidueName()] = cycle.mCycleFrame[cycle.getCycleResidueName()]
                if(self.mOptions.mDebug):
                    cycle.check_not_nan(self.mCycleFrame[cycle.getCycleResidueName()].values ,
                                        cycle.getCycleResidueName())
                end_time = time.time()
                lTrainingTime = round(end_time - start_time , 2);
                if(self.mOptions.mDebugProfile):
                    logger = tsutil.get_pyaf_logger();
                    logger.info("CYCLE_TRAINING_TIME_IN_SECONDS '" + cycle.mOutName + "' " + str(lTrainingTime))
        pass


    def filterSeasonals(self):
        logger = tsutil.get_pyaf_logger();
        logger.debug("CYCLE_TRAINING_FILTER_SEASONALS_START")
        for trend in self.mTrendList:
            lPerfs = {}
            lTrend_residue_name = trend.mOutName + '_residue'
            lCycleList = []
            lSeasonals = {}
            for cycle in self.mCycleList[trend]:
                if(isinstance(cycle , cSeasonalPeriodic)):
                    cycle.computePerf();
                    # check that the MAPE is not above 1.0
                    if(cycle.mCycleForecastPerf.is_acceptable_criterion_value(self.mOptions.mCycle_Criterion)):
                        lCritValue = cycle.mCycleForecastPerf.getCriterionValue(self.mOptions.mCycle_Criterion)
                        lCategories = len(cycle.mEncodedValueDict.keys())
                        lPerfs[cycle.mOutName] = (round(lCritValue, 3), lCategories)
                        lSeasonals[cycle.mOutName] = cycle
                else:
                    lCycleList = lCycleList + [cycle]
            
            if(len(lSeasonals) == 0):
                return
            lData = lPerfs.items()
            # less MAPE is better, less categories is better, the last is the name of the seasonal to have a total order.
            lSortingMethod_By_MAPE = lambda x : (x[1][0], x[0])
            lData = sorted(lData, key = lSortingMethod_By_MAPE)
            assert(len(lData) > 0)
            lBestPerf = lSeasonals[ lData[0][0] ].mCycleForecastPerf
            lBestCriterion = lData[0][1]
            lData_smallest = [x for x in lData if lBestPerf.is_close_criterion_value(self.mOptions.mCycle_Criterion,
                                                                                     x[1][0],
                                                                                     iTolerance = 0.05)]
            lSortingMethod_By_Complexity = lambda x : (x[1][1], x[0])
            lData_smallest = sorted(lData_smallest, key = lSortingMethod_By_Complexity)
            assert(len(lData_smallest) > 0)
            lBestSeasonal = lSeasonals[ lData_smallest[0][0] ]
            lBestCriterion = lData_smallest[0][1]
            lCycleList = lCycleList + [lBestSeasonal]
            self.mCycleList[trend] = lCycleList
            if(self.mOptions.mDebugCycles):
                logger.info("CYCLE_TRAINING_FILTER_SEASONALS_DATA " + trend.mOutName + " " + str(lData_smallest))
                logger.info("CYCLE_TRAINING_FILTER_SEASONALS_BEST " + trend.mOutName + " " + lBestSeasonal.mOutName + " " + str(lBestCriterion))
            logger.debug("CYCLE_TRAINING_FILTER_SEASONALS_END")
        pass

    def estimateAllCycles(self):
        self.defineCycles();
        self.estimateCycles()
        if(self.mOptions.mFilterSeasonals):
            self.filterSeasonals()
        
