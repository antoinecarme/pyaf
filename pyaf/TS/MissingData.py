# Copyright (C) 2016 Antoine Carme <Antoine.Carme@outlook.com>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license



import pandas as pd
import numpy as np
import datetime

# this is a specialized missing data imputer for the time and signal
# it performs a simple interpolation (for the moment)

class cMissingDataImputer:

    def __init__(self):
        self.mOptions = None;


    def has_missing_data(self, iSeries):
        return iSeries.isnull().values.any()

    def apply(self, iInputDS, iTime, iSignal):
        iInputDS = self.apply_time_imputation_method(iInputDS, iTime)
        iInputDS = self.apply_signal_imputation_method(iInputDS, iSignal)
        return iInputDS

    def apply_time_imputation_method(self, iInputDS, iTime):
        if(self.mOptions.mMissingDataOptions.mTimeMissingDataImputation is None):
            return iInputDS
        
        if(not self.has_missing_data(iInputDS[iTime])):
            return iInputDS
        
        elif(self.mOptions.mMissingDataOptions.mTimeMissingDataImputation == "DiscardRow"):
            iInputDS = iInputDS[iInputDS[iTime].notnull()]

        elif(self.mOptions.mMissingDataOptions.mTimeMissingDataImputation == "Interpolate"):
            lTime = self.interpolate_time_if_needed(iInputDS , iTime)
            iInputDS[iTime] = lTime

        return iInputDS
        
    def apply_signal_imputation_method(self, iInputDS, iSignal):
        if(self.mOptions.mMissingDataOptions.mSignalMissingDataImputation is None):
            return iInputDS
            
        if(not self.has_missing_data(iInputDS[iSignal])):
            return iInputDS
        elif(self.mOptions.mMissingDataOptions.mSignalMissingDataImputation == "DiscardRow"):
            iInputDS = iInputDS[iInputDS[iSignal].notnull()]

        elif(self.mOptions.mMissingDataOptions.mSignalMissingDataImputation == "Interpolate"):
            lSignal = self.interpolate_signal_if_needed(iInputDS , iSignal)
            iInputDS[iSignal] = lSignal
            
        elif(self.mOptions.mMissingDataOptions.mSignalMissingDataImputation == "Constant"):
            lSignal = iInputDS[iSignal].fillna(self.mOptions.mMissingDataOptions.mConstant, method=None)
            iInputDS[iSignal] = lSignal
            
        elif(self.mOptions.mMissingDataOptions.mSignalMissingDataImputation == "Mean"):
            lMean = iInputDS[iSignal].mean()
            lSignal = iInputDS[iSignal].fillna(lMean, method=None)
            iInputDS[iSignal] = lSignal
            
        elif(self.mOptions.mMissingDataOptions.mSignalMissingDataImputation == "Median"):
            lMedian = iInputDS[iSignal].median()
            lSignal = iInputDS[iSignal].fillna(lMedian, method=None)
            iInputDS[iSignal] = lSignal
            
        elif(self.mOptions.mMissingDataOptions.mSignalMissingDataImputation == "PreviousValue"):
            lSignal = iInputDS[iSignal].fillna(method='ffill')
            # replace the first empty values with the first known value
            lSignal = lSignal.fillna(method='bfill')
            iInputDS[iSignal] = lSignal
            
        return iInputDS

    def interpolate_time_if_needed(self, iInputDS , iTime):
        
        type1 = iInputDS[iTime].dtype
        if(type1.kind == 'M'):
            lMin = iInputDS[iTime].min()
            lDiffs = iInputDS[iTime] - lMin
            lDiffs = lDiffs.apply(lambda x : x.total_seconds())
            # print("TIME_MIN" , lMin)
            # print("TIME" , iInputDS[iTime].describe())
            # print("TIME_DIFFS" , lDiffs.describe())
            lTime = lDiffs.interpolate(method='linear', limit_direction='both', axis=0)
            lTime = lTime.apply(lambda x : lMin + datetime.timedelta(seconds=x))
            # print("TIME2" , lTime.describe())
            lTime = lTime.astype(type1)
            return lTime
        else:
            lTime = iInputDS[iTime].interpolate(method='linear', limit_direction='both', axis=0)
            lTime = lTime.astype(type1)
            return lTime
            
    def interpolate_signal_if_needed(self, iInputDS , iSignal):
        lSignal = iInputDS[iSignal].interpolate(method='linear', limit_direction='both', axis=0)
        lSignal = lSignal.astype(iInputDS[iSignal].dtype)
        return lSignal
