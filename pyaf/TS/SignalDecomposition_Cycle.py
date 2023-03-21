# Copyright (C) 2016 Antoine Carme <Antoine.Carme@outlook.com>
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
from . import Complexity as tscomplex

class cAbstractCycle:
    def __init__(self , trend):
        self.mTimeInfo = tsti.cTimeInfo()
        self.mTrendFrame = None
        self.mCycleFrame = None
        self.mTrend = trend;
        self.mTrend_residue_name = self.mTrend.mOutName + '_residue'
        self.mFormula = None;
        self.mComplexity = tscomplex.eModelComplexity.High;
    

    def getCycleResidueName(self):
        return self.getCycleName() + "_residue";


    def plot(self):
        tsplot.decomp_plot(self.mCycleFrame, self.mTimeInfo.mNormalizedTimeColumn,
                           self.mTrend_residue_name, self.getCycleName() , self.getCycleResidueName(), horizon = self.mTimeInfo.mHorizon);


    def check_not_nan(self, sig , name):
        #print("check_not_nan");
        if(np.isnan(sig[:-1]).any() or np.isinf(sig[:-1]).any() ):
            logger = tsutil.get_pyaf_logger();
            logger.error("CYCLE_RESIDUE_WITH_NAN_IN_SIGNAL" + str(sig));
            raise tsutil.Internal_PyAF_Error("CYCLE_COLUMN _FOR_TREND_RESIDUE ['"  + name + "'");
        # print("check_cycle_residue ", (name, self.mDecompositionType, sig[:-1].min(), sig[:-1].max(), sig[:-1].mean(), sig[:-1].std()))
        if(sig[:-1].max() > 1e5):
            self.dump_values()
            raise tsutil.Internal_PyAF_Error("Invalid cycle_residue_too_large '" + str(name) + "'");
        pass
        pass

    def compute_cycle_residue(self, df):
        lSignal = df[self.mSignal]
        lTrend = df[self.mTrend.mOutName]
        lCycle = df[self.getCycleName()]
        lOutName = self.getCycleName()
        if(self.mDecompositionType in ['T+S+R']):
            df[self.getCycleResidueName()] = lSignal - lTrend - lCycle
        elif(self.mDecompositionType in ['TS+R']):
            df[self.getCycleResidueName()] = lSignal - lTrend * lCycle 
        else:
            lTrendCycle = lTrend * lCycle
            # This is questionable. But if only a few values are zero, it is the safest.
            lTrendCycle = lTrendCycle.apply(lambda x : x if(abs(x) > 1e-2) else 1e-2)
            df[self.getCycleResidueName()] = lSignal / lTrendCycle
        # df_detail = df[[self.mSignal, self.mTrend.mOutName, self.getCycleName(), self.getCycleResidueName()]]
        # print("compute_cycle_residue_detail ", (lOutName, self.mDecompositionType, df_detail.describe(include='all').to_dict()))
        

    def compute_target_means_by_cycle_value(self , iCycleFrame, iCycleName):
        # we encode only using estimation
        lCycleFrameEstim = self.mSplit.getEstimPart(iCycleFrame);
        lGroupBy = lCycleFrameEstim.groupby(by=[iCycleName] , sort=False)[self.mTrend_residue_name]
        lEncodedValueDict = None
        if(self.mOptions.mCycle_Encoding_Scheme == "Target_Mean"):
            lEncodedValueDict = lGroupBy.mean().to_dict();
        else:
            lEncodedValueDict = lGroupBy.median().to_dict();
        for x in lEncodedValueDict.keys():
            lEncodedValueDict[ x ] = np.float64(lEncodedValueDict[ x ])
        return lEncodedValueDict

    def compute_target_means_default_value(self):
        # we encode only using estimation
        lCycleFrameEstim = self.mSplit.getEstimPart(self.mCycleFrame);
        if(self.mOptions.mCycle_Encoding_Scheme == "Target_Mean"):
            return np.float64(lCycleFrameEstim[self.mTrend_residue_name].mean());
        return np.float64(lCycleFrameEstim[self.mTrend_residue_name].median());

    def computePerf(self):
        if(self.mOptions.mDebug):
            self.check_not_nan(self.mCycleFrame[self.getCycleResidueName()], self.getCycleResidueName())
        # self.mCycleFrame.to_csv(self.getCycleResidueName() + ".csv");
        self.mCycleFitPerf = tsperf.cPerf();
        self.mCycleForecastPerf = tsperf.cPerf();
        # self.mCycleFrame[[self.mTrend_residue_name, self.getCycleName()]].to_csv(self.getCycleName() + ".csv");
        (lFrameFit, lFrameForecast, lFrameTest) = self.mSplit.cutFrame(self.mCycleFrame);
        
        self.mCycleFitPerf.computeCriterionValues(
            lFrameFit[self.mTrend_residue_name], lFrameFit[self.getCycleName()],
            [self.mOptions.mCycle_Criterion], self.getCycleName())
        self.mCycleForecastPerf.computeCriterionValues(
            lFrameForecast[self.mTrend_residue_name], lFrameForecast[self.getCycleName()],
            [self.mOptions.mCycle_Criterion], self.getCycleName())
        self.dumpCyclePerf()

    
    def dumpCyclePerf(self):
        if(self.mOptions.mDebugCycles):
            logger = tsutil.get_pyaf_logger();
            lDict = {
                "Fit" : self.mCycleFitPerf.getCriterionValue(self.mOptions.mCycle_Criterion),
                "Forecast" : self.mCycleForecastPerf.getCriterionValue(self.mOptions.mCycle_Criterion),
            }
            logger.info("CYCLE_PERF_DETAIL " + self.mOptions.mCycle_Criterion  + " " + self.mOutName +  " " + str(lDict))


class cZeroCycle(cAbstractCycle):

    def __init__(self , trend):
        super().__init__(trend);
        self.mFormula = "NoCycle"
        self.mComplexity = tscomplex.eModelComplexity.Low;
        self.mConstantValue = 0.0

    def getCycleName(self):
        return self.mTrend_residue_name + "_zeroCycle[" + str(self.mConstantValue) + "]";


    def dump_values(self):
        logger = tsutil.get_pyaf_logger();
        logger.info("ZERO_CYCLE_MODEL_VALUES " + self.getCycleName() + " " + str(self.mConstantValue) + " {}");        
    
    def fit(self):
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mTimeInfo.addVars(self.mCycleFrame);
        self.mConstantValue = 0.0        
        if(self.mDecompositionType in ['TS+R', 'TSR']):
            # multiplicative models
            self.mConstantValue = 1.0

        self.mCycleFrame[self.mTrend.mOutName] = self.mTrendFrame[self.mTrend.mOutName]
        self.mCycleFrame[self.mTrend_residue_name] = self.mTrendFrame[self.mTrend_residue_name]
        self.mCycleFrame[self.getCycleName()] = self.mConstantValue
        self.mOutName = self.getCycleName()
        self.compute_cycle_residue(self.mCycleFrame)
        
    def transformDataset(self, df):
        df[self.getCycleName()] = self.mConstantValue;
        self.compute_cycle_residue(df)
        return df;

class cSeasonalPeriodic(cAbstractCycle):
    def __init__(self , trend, date_part):
        super().__init__(trend);
        self.mDatePart = date_part;
        self.mEncodedValueDict = {}
        self.mFormula = "Seasonal_" + self.mDatePart.name;
        

    def getCycleName(self):
        return self.mTrend_residue_name + "_Seasonal_" + self.mDatePart.name;

    def dump_values(self):
        logger = tsutil.get_pyaf_logger();
        lDict = dict(sorted([(k, round(v, 6)) for (k, v) in self.mEncodedValueDict.items()]))
        logger.info("SEASONAL_MODEL_VALUES " + self.getCycleName() + " " + str(round(self.mDefaultValue, 6)) + " " + str(lDict));


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
        self.mCycleFrame[self.mTrend.mOutName] = self.mTrendFrame[self.mTrend.mOutName]
        self.mCycleFrame[lName] = self.compute_date_parts(self.mTrendFrame[self.mTime])
        self.mDefaultValue = self.compute_target_means_default_value()
        self.mEncodedValueDict = self.compute_target_means_by_cycle_value(self.mCycleFrame, self.getCycleName())

        self.mCycleFrame[lName + '_enc'] = self.mCycleFrame[lName].map(self.mEncodedValueDict).fillna(self.mDefaultValue)
        self.mCycleFrame[lName + '_enc'].fillna(self.mDefaultValue, inplace=True);
        self.mCycleFrame[lName + '_NotEncoded'] = self.mCycleFrame[lName];
        self.mCycleFrame[lName] = self.mCycleFrame[lName + '_enc'];
        
        self.mOutName = self.getCycleName()
        #print("encoding '" + lName + "' " + str(self.mEncodedValueDict));
        # The longer the seasonal, the more complex it is.
        self.mComplexity = tscomplex.eModelComplexity.Low;
        if(len(self.mEncodedValueDict.keys()) > 31):
            self.mComplexity = tscomplex.eModelComplexity.High;
        self.compute_cycle_residue(self.mCycleFrame)

    def transformDataset(self, df):
        lDateParts = self.compute_date_parts(df[self.mTime])
        df[self.getCycleName()] = lDateParts.map(self.mEncodedValueDict).fillna(self.mDefaultValue)
        self.compute_cycle_residue(df)
        return df;

class cBestCycleForTrend(cAbstractCycle):
    def __init__(self , trend, criterion):
        super().__init__(trend);
        self.mCycleFrame = None
        self.mCyclePerfByLength = {}
        self.mBestCycleValueDict = {}
        self.mBestCycleLength = None
        self.mCriterion = criterion
        self.mFormula = "BestCycle"
        
    def getCycleName(self):
        return self.mTrend_residue_name + "_Cycle_" + str(self.mBestCycleLength)            

    def dump_values(self):
        logger = tsutil.get_pyaf_logger();
        lDict = {} if(self.mBestCycleLength is None) else self.mBestCycleValueDict[self.mBestCycleLength]
        lDict = dict(sorted([(k, round(v, 6)) for (k, v) in lDict.items()]))
        logger.info("BEST_CYCLE_LENGTH_VALUES " + self.getCycleName() + " " + str(self.mBestCycleLength) + " " + str(round(self.mDefaultValue, 6)) + " " + str(lDict));

    
    def computeBestCycle(self):
        self.mBestCycleLength = None;
        lData = self.mCyclePerfByLength.items()
        if(len(lData) == 0):
            return
        
        lPerf = tsperf.cPerf();
        # less MAPE is better, less categories is better, the last is the length to have a total order.
        lSortingMethod_By_MAPE = lambda x : (x[1][0], x[0])
        lData = sorted(lData, key = lSortingMethod_By_MAPE) # MAPE => MASE
        assert(len(lData) > 0)
        lBestCriterion = lData[0][1]
        lData_smallest = [x for x in lData if lPerf.is_close_criterion_value(self.mOptions.mCycle_Criterion,
                                                                             x[1][0],
                                                                             iTolerance = 0.05, iRefValue = lBestCriterion[0])]
        lSortingMethod_By_Length = lambda x : (x[1][1], x[0])
        lData_smallest = sorted(lData_smallest, key = lSortingMethod_By_Length)
        assert(len(lData_smallest) > 0)
        self.mBestCycleLength = lData_smallest[0][0]

        # self.transformDataset(self.mCycleFrame);
        pass

    def get_tested_cycles(self, signal_length):
        if(self.mOptions.mCycleLengths is None):
            lAbsMax = 366
            N = min(signal_length // 6, lAbsMax)
            lCycleLengths0 = [x for x in range(3, 60)]
            lCycleLengths1 = [5, 7, 12, 24, 30, 60]
            lCycleLengths2 = [x*k for x in lCycleLengths1 for k in range(2, N // 2) if((x*k) <= N)]
            lCycleLengths3 = lCycleLengths0 + lCycleLengths1 + lCycleLengths2
            lCycleLengths4 = list(set(lCycleLengths3))
            return [x for x in lCycleLengths4 if (x < signal_length // 6)]
        return [x for x in self.mOptions.mCycleLengths if (x < signal_length // 6)]

    def generate_cycles(self):
        self.mTimeInfo.addVars(self.mCycleFrame);
        self.mCycleFrame[self.mTrend_residue_name] = self.mTrendFrame[self.mTrend_residue_name]
        self.mCycleFrame[self.mTrend.mOutName] = self.mTrendFrame[self.mTrend.mOutName]
        self.mDefaultValue = self.compute_target_means_default_value();
        self.mCyclePerfByLength = {}
        lEstimResidue = self.mSplit.getEstimPart(self.mTrendFrame[self.mTrend_residue_name])
        lCycleLengths = self.get_tested_cycles(lEstimResidue.shape[0])
        lTimer = None
        if(self.mOptions.mDebugCycles):
            lTimer = tsutil.cTimer(("SELECTING_BEST_CYCLE_FOR_MODEL", self.mTrend_residue_name, len(lCycleLengths), lCycleLengths))
        lCycleFrame = pd.DataFrame(index = self.mTrendFrame.index);
        lCycleFrame[self.mTrend_residue_name ] = self.mTrendFrame[self.mTrend_residue_name]
        for lLength in lCycleLengths:
            if (lLength > 1):
                name_length = self.mTrend_residue_name + '_Cycle';
                lCycleFrame[name_length] = self.mCycleFrame[self.mTimeInfo.mRowNumberColumn] % lLength
                lEncodedValueDict = self.compute_target_means_by_cycle_value(lCycleFrame, name_length)
                lCycleFrame[name_length + '_enc'] = lCycleFrame[name_length].map(lEncodedValueDict).fillna(self.mDefaultValue)

                self.mBestCycleValueDict[lLength] = lEncodedValueDict;
                
                lPerf = tsperf.cPerf();
                # validate the cycles on the validation part
                lValidFrame = self.mSplit.getValidPart(lCycleFrame);
                lCritValues = lPerf.computeCriterionValues(lValidFrame[self.mTrend_residue_name],
                                                    lValidFrame[name_length + "_enc"],
                                                    [self.mCriterion],
                                                    "Validation")
                lCritValue = lCritValues[self.mCriterion]
                if(lCritValue is None):
                    print("CYCFLE_CRITERION_VALUES", lCritValues)
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
            self.mFormula = "Cycle_" + str(self.mBestCycleLength);
            
        self.transformDataset(self.mCycleFrame);
        self.mComplexity = tscomplex.eModelComplexity.Low;
        if(self.mBestCycleLength is not None):
            lDict = self.mBestCycleValueDict[self.mBestCycleLength];
            if(len(lDict.keys()) > 31):
                self.mComplexity = tscomplex.eModelComplexity.High;

    def transformDataset(self, df):
        if(self.mBestCycleLength is not None):
            lValueCol = df[self.mTimeInfo.mRowNumberColumn].mod(self.mBestCycleLength);
            df['cycle_internal'] = lValueCol;
            # print("BEST_CYCLE" , self.mBestCycleLength)
            # print(self.mBestCycleValueDict);
            lDict = self.mBestCycleValueDict[self.mBestCycleLength];
            df[self.getCycleName()] = lValueCol.map(lDict).fillna(self.mDefaultValue)
        else:
            df[self.getCycleName()] = np.zeros_like(df[self.mTimeInfo.mRowNumberColumn]);
            if(self.mDecompositionType in ['TS+R', 'TSR']):
                # multiplicative models
                df[self.getCycleName()] = 1.0

        self.compute_cycle_residue(df)
        if(self.mOptions.mDebug):
            self.check_not_nan(df[self.getCycleName()].values , self.getCycleName());

        return df;

class cCycleEstimator:
    
    def __init__(self):
        self.mTimeInfo = tsti.cTimeInfo()
        self.mTrendFrame = None
        self.mCycleFrame = None
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
            lTrend_residue_name = trend.mOutName + '_residue'

            if(self.mOptions.mActivePeriodics['NoCycle']):
                self.mCycleList[trend] = [cZeroCycle(trend)];
            lThreshold = 0.001 # The signal is scaled to be between 0 and 1
            lEstimResidue = self.mSplit.getEstimPart(self.mTrendFrame[lTrend_residue_name])
            lTrendRange = lEstimResidue.max() - lEstimResidue.min()
            lKeep = (lEstimResidue.shape[0] >= 12) and (lTrendRange >= lThreshold) # Keep this test as simple as possible.
            if(lKeep and self.mOptions.mActivePeriodics['BestCycle']):
                self.mCycleList[trend] = self.mCycleList[trend] + [
                    cBestCycleForTrend(trend, self.mOptions.mCycle_Criterion)];
            if(lKeep and self.mTimeInfo.isPhysicalTime()):
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
                cycle.mCycleFrame = pd.DataFrame(index = self.mTrendFrame.index)
                cycle.mTimeInfo = self.mTimeInfo;
                cycle.mSplit = self.mSplit;
                cycle.mOptions = self.mOptions;
                cycle.mDecompositionType = self.mDecompositionType
            
    def plotCycles(self):
        for trend in self.mTrendList:
            for cycle in self.mCycleList[trend]:
                cycle.plot()

    


    def estimateCycles(self):
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mCycleFrame = pd.DataFrame(index = self.mTrendFrame.index);
        self.mTimeInfo.addVars(self.mCycleFrame);
        for trend in self.mTrendList:
            lTrend_residue_name = trend.mOutName + '_residue'
            self.mCycleFrame[lTrend_residue_name] = self.mTrendFrame[lTrend_residue_name]
            for cycle in self.mCycleList[trend]:
                cycle.fit();
                cycle.computePerf();
                self.mCycleFrame[cycle.getCycleName()] = cycle.mCycleFrame[cycle.getCycleName()]
                self.mCycleFrame[cycle.getCycleResidueName()] = cycle.mCycleFrame[cycle.getCycleResidueName()]
                if(self.mOptions.mDebug):
                    cycle.check_not_nan(self.mCycleFrame[cycle.getCycleResidueName()].values ,
                                        cycle.getCycleResidueName())
            # Avoid dataframe fragmentation warnings.
            self.mCycleFrame = self.mCycleFrame.copy()
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
            lData = sorted(lData, key = lSortingMethod_By_MAPE) # MAPE => MASE
            assert(len(lData) > 0)
            lBestPerf = lSeasonals[ lData[0][0] ].mCycleForecastPerf
            lBestCriterion = lData[0][1]
            lData_smallest = [x for x in lData if lBestPerf.is_close_criterion_value(self.mOptions.mCycle_Criterion,
                                                                                     x[1][0],
                                                                                     iTolerance = 0.05)]
            lSortingMethod_By_Length = lambda x : (x[1][1], x[0])
            lData_smallest = sorted(lData_smallest, key = lSortingMethod_By_Length)
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
        lTimer = None
        if(self.mOptions.mDebugProfile):
            lTimer = tsutil.cTimer(("TRAINING_CYCLES", {"Signal" : self.mTimeInfo.mSignal}))
        self.defineCycles();
        self.estimateCycles()
        if(self.mOptions.mFilterSeasonals):
            self.filterSeasonals()
        
        for trend in self.mTrendList:
            for cycle in self.mCycleList[trend]:
                del cycle.mCycleFrame
