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
        
    def tuple_to_string(self, k):
        str1 = "_".join(list(k));
        # print(k , "=>" , str1);
        return str1;
    
    def add_level(self, previous_level):
        level = previous_level + 1;
        self.mStructure[level] = {};
        for group in self.mStructure[previous_level]:
            lGroupLabel = group; # self.tuple_to_string(group);
            lTuple = self.mLabels2Tuples[lGroupLabel]
            for k in [previous_level]:
                if(lTuple[k] != ""):
                    new_group = list(lTuple);
                    new_group[k] = "";
                    new_group = tuple(new_group);
                    lNewGroupLabel = self.tuple_to_string(new_group);
                    self.mLabels2Tuples[lNewGroupLabel] = new_group;
                    if(lNewGroupLabel not in self.mStructure[level]):
                        self.mStructure[level][lNewGroupLabel] = set();
                    self.mStructure[level][lNewGroupLabel].add(lGroupLabel)
        # print("STRUCTURE_LEVEL" , level, self.mStructure[level]);


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
        
    def create_HierarchicalStructure__(self):
        
        self.define_groups()
        lGroups = self.mGroups
        self.mLevels = list(lGroups.keys());
        self.mLabels2Tuples = {};
        self.mStructure = {};
        array1 = [ sorted(lGroups[k]) for k in self.mGroupOrder ];
        prod = itertools.product( *array1 );
        # print(prod);
        level = 0;
        self.mStructure[level] = {}
        lSignal = self.mSignal
        for k in prod:
            # print("PRODUCT_DETAIL", k);
            lGroupLabel = self.tuple_to_string(k);
            self.mLabels2Tuples[lGroupLabel] = k;
            self.mStructure[level][lGroupLabel] = set();
        # print("STRUCTURE_LEVEL" , level, self.mStructure[level]);
        while(len(self.mStructure[level]) > 1):
            self.add_level(level);
            level = level + 1;
        
        # print("STRUCTURE", self.mStructure);
        pass


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
