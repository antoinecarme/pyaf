# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np
import itertools

from . import DateTime_Functions as dtfunc
from . import SignalHierarchy as sighier


class cTemporalHierarchy (sighier.cSignalHierarchy):

    def __init__(self):
        sighier.cSignalHierarchy.__init__(self)
        self.mLabels2Tuples = {};
        

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
        df = self.add_temporal_data(df)
        return df
    
    def add_temporal_data(self, df):
        print(df.head())
        N = len(df.columns)
        lDate = self.mTrainingDataset[self.mDateColumn]
        lPrefix = "TH"
        df[lPrefix + '_dayOfMonth'] = lDate.dt.day
        df[lPrefix + '_dayname'] = lDate.dt.day_name()
        df[lPrefix + '_D'] = lDate.dt.weekday
        df[lPrefix + '_M'] = lDate.dt.month
        df[lPrefix + '_W'] = lDate.dt.week
        df[lPrefix + '_weekOfQuarter'] = lDate.dt.week % 13
        df[lPrefix + '_Q'] = lDate.dt.quarter
        df[lPrefix + '_HalfYear'] = lDate.dt.quarter // 2
        df[lPrefix + '_Y'] = lDate.dt.year
        
        print(df.head())
        for lPeriod in self.mPeriods:
            lName = lPrefix + "_" + lPeriod + "_start"
            df[lName] = lDate.apply(self.get_beginning_of_period , args=(lPeriod))
            WData = df.groupby([lName])[self.mSignal].sum().reset_index()
            df_merge = df[[self.mDateColumn , self.mSignal]].merge(WData, left_on=self.mDateColumn,right_on=lName, how='left', suffixes=('_x', '_Period'))
            df[self.mSignal + '_' + lPeriod] = df_merge[self.mSignal + '_Period']
        return df

    def define_groups__(self):
        lPrefix = "TH_"
        self.mGroups = {}
        for lPeriod in self.mPeriods:
            self.mGroups[lPeriod] = [self.mSignal + '_' + lPeriod]
        self.mGroupOrder= [lPeriod for lPeriod in self.mPeriods]
        
    def create_HierarchicalStructure(self):
        self.mPeriods = self.mHierarchy['Periods']
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
