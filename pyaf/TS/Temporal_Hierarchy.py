# Copyright (C) 2016 Antoine Carme <Antoine.Carme@outlook.com>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np

from . import DateTime_Functions as dtfunc
from . import SignalHierarchy as sighier
from . import Utils as tsutil


class cTemporalHierarchy (sighier.cSignalHierarchy):

    def __init__(self):
        sighier.cSignalHierarchy.__init__(self)
        self.mHorizons = {}
        

    def discard_nans_in_aggregate_signals(self):
        return True

    def get_specific_date_column_for_signal(self, level, signal):
        # only for temporal hierarchies
        lPeriod = self.mPeriods[level]
        lPrefix = "TH"
        lName = lPrefix + "_" + lPeriod + "_start"
        return lName
    
    def get_beginning_of_period(self, x, iPeriod):
        # add this utility function.
        lHelper = dtfunc.cDateTime_Helper()
        return lHelper.get_beginning_of_period(iPeriod, x)


    def aggregate_time_columns(self, level, signal, iAllLevelsDataset):
        cols = [col1 for col1 in sorted(self.mStructure[level][signal])]
        iAllLevelsDataset[signal] = iAllLevelsDataset[cols[0]]
        for col in cols[1:]:
            # logical or
            new_col  = iAllLevelsDataset[[signal, col]].apply(lambda x : x[1] if (x[0] is None) else x[0], axis = 0)
            iAllLevelsDataset[signal] = new_col
    
    def create_all_levels_dataset(self, df):
        df = df.reset_index(drop = True)
        df = self.add_temporal_data(df)
        return df

    def get_horizon(self, level, signal):
        # only for temporal hierarchies
        lPeriod = self.mPeriods[level]
        return self.mHorizons[lPeriod]

    
    def add_temporal_data(self, df):
        logger = tsutil.get_pyaf_hierarchical_logger();
        N = len(df.columns)
        df1 = df[[self.mDateColumn, self.mSignal]].copy()
        df1.set_index(self.mDateColumn, inplace=True, drop=False)
        
        # df1.info()
        lPrefix = "TH"
        df_resampled = {}
        for lPeriod in self.mPeriods:
            lName = lPrefix + "_" + lPeriod + "_start"
            df_resampled[lPeriod] = df1[self.mSignal].resample(lPeriod).sum().reset_index()
            df_resampled[lPeriod].columns = [lName , self.mSignal]
            # synchronize
            lShift = df_resampled[lPeriod][lName].iloc[0] - df[self.mDateColumn].iloc[0] 
            df_resampled[lPeriod][lName] = df_resampled[lPeriod][lName] - lShift
            
        for lPeriod in self.mPeriods:
            lName = lPrefix + "_" + lPeriod + "_start"
            WData = df_resampled[lPeriod]
            # df[[self.mDateColumn , self.mSignal]].info()
            # WData.info()
            # print("DATE_PERIOD", list(WData[lName])[:30])
            df_merge = df[[self.mDateColumn , self.mSignal]].merge(WData, left_on=self.mDateColumn,right_on=lName, how='left', suffixes=('_x', '_Period'), sort=True)
            df[self.mSignal + '_' + lPeriod] = df_merge[self.mSignal + '_Period']
            df[lName] = df_merge[lName]
            logger.info("FORECASTING_HIERARCHICAL_TEMPORAL_LEVEL " + str((lPeriod, lName, list(df.columns), WData.shape)))
            

        return df

    def compute_horizons(self, df):
        df = df.reset_index(drop = True)
        logger = tsutil.get_pyaf_hierarchical_logger();
        N = len(df.columns)
        df1 = df[[self.mDateColumn, self.mSignal]].copy()
        df1.set_index(self.mDateColumn, inplace=True, drop=False)
        
        # df1.info()
        lPrefix = "TH"
        lHelper = dtfunc.cDateTime_Helper()
        lBaseFreq = lHelper.computeTimeFrequency_in_seconds(df1[self.mDateColumn])
        df_resampled = {}
        for lPeriod in self.mPeriods:
            lName = lPrefix + "_" + lPeriod + "_start"
            df_resampled[lPeriod] = df1[self.mSignal].resample(lPeriod).sum().reset_index()
            df_resampled[lPeriod].columns = [lName , self.mSignal]
            # synchronize
            lShift = df_resampled[lPeriod][lName].iloc[0] - df[self.mDateColumn].iloc[0] 
            df_resampled[lPeriod][lName] = df_resampled[lPeriod][lName] - lShift
            lDate_Period = df_resampled[lPeriod][lName]
            # print("AS_FREQ" , lPeriod , lDate_Period.head())
            lNewFreq = lHelper.computeTimeFrequency_in_seconds(lDate_Period)
            lHorizon = int(self.mHorizon * lBaseFreq / lNewFreq)
            lHorizon = max(1, lHorizon)
            # print("AS_FREQ_2" , lPeriod , lBaseFreq , lNewFreq , lHorizon)
            self.mHorizons[lPeriod] = lHorizon
            logger.info("FORECASTING_HIERARCHICAL_TEMPORAL_HORIZONS_FIRST_RESAMPLED_DATA " + str(lPeriod) + " " + str(df_resampled[lPeriod].head(5).to_dict()) )

        logger.info("FORECASTING_HIERARCHICAL_TEMPORAL_HORIZONS " + str(self.mHorizons));

    def checkPhysicalTime(self):
        logger = tsutil.get_pyaf_hierarchical_logger();
        lHelper = dtfunc.cDateTime_Helper()
        lIsPhysical = lHelper.isPhysicalTime(self.mTrainingDataset[self.mDateColumn])
        if(not lIsPhysical):
            raise tsutil.PyAF_Error('TIME_HIERARCHY_PHYSICAL_TIME_NEEDED ' + str(self.mDateColumn)  + " " + str(self.mTrainingDataset[self.mDateColumn].dtype))
        
    def check_increasing_periods(self):
        logger = tsutil.get_pyaf_hierarchical_logger();
        lHelper = dtfunc.cDateTime_Helper()
        lSeconds = {}
        for lPeriod in self.mPeriods:
            lSeconds[lPeriod] = lHelper.get_period_length_in_seconds(lPeriod)
        logger.info("FORECASTING_HIERARCHICAL_TEMPORAL_FREQUENCIES " + str(lSeconds) )
        lPreviousPeriod = lSeconds[ self.mPeriods[0] ]
        
        lTimeFreqInSeconds = lHelper.computeTimeFrequency_in_seconds(self.mTrainingDataset[self.mDateColumn])
        if(lTimeFreqInSeconds > lPreviousPeriod):
            raise tsutil.PyAF_Error('TIME_HIERARCHY_PHYSICAL_TIME_RESOLUTION_TOO_LOW_FOR_THIS_PERIOD ' + str(self.mDateColumn)  + " " + str(lTimeFreqInSeconds) +
                                    " " + self.mPeriods[0] + " " +  str(lPreviousPeriod))
        
        for lPeriod in self.mPeriods[1:]:
            if(lSeconds[lPeriod] < lPreviousPeriod):
                raise tsutil.PyAF_Error('TIME_HIERARCHY_NOT_MONOTONOUS ' + str(self.mPeriods));
            lPreviousPeriod = lSeconds[lPeriod]
        
    def create_HierarchicalStructure(self):
        self.mPeriods = self.mHierarchy['Periods']
        self.checkPhysicalTime()
        self.check_increasing_periods()
        self.compute_horizons(self.mTrainingDataset)
        # self.add_temporal_data(self.mTrainingDataset)
        self.mLevels = list(range(len(self.mPeriods)));
        self.mStructure = {};
        for (lLevel, lPeriod) in enumerate(self.mPeriods):
            self.mStructure[lLevel] = {}
            
        for (lLevel, lPeriod) in enumerate(self.mPeriods):
            self.mStructure[lLevel][self.mSignal + '_' + lPeriod] = set()
            if(lLevel > 0):
                self.mStructure[lLevel][self.mSignal + '_' + lPeriod] = set([self.mSignal + '_' + self.mPeriods[lLevel - 1]])
                
        # print(self.mStructure);
        pass
