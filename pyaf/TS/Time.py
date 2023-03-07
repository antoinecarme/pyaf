# Copyright (C) 2016 Antoine Carme <Antoine.Carme@outlook.com>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np
from enum import IntEnum

from . import Utils as tsutil
from . import TimeSeries_Cutting as tscut
from . import DateTime_Functions as dtfunc
    
class cTimeInfo:
    # class data

    def __init__(self):
        self.mSignalFrame = None
        self.mTimeMin = None;
        self.mTimeMax = None;
        self.mTimeMinMaxDiff = None;
        self.mTimeDelta = None;
        self.mHorizon = None;        
        self.mResolution = dtfunc.eTimeResolution.NONE
        self.mSplit = None

    def info(self):
        lStr2 = "TimeVariable='" + self.mTime +"'";
        lStr2 += " TimeMin=" + str(self.mTimeMin) +"";
        lStr2 += " TimeMax=" + str(self.mTimeMax) +"";
        lStr2 += " TimeDelta=" + str(self.mTimeDelta) +"";
        lStr2 += " Horizon=" + str(self.mHorizon) +"";
        return lStr2;


    def to_dict(self):
        dict1 = {};
        dict1["TimeVariable"] =  self.mTime;
        dict1["TimeMin"] =  str(self.mSignalFrame[self.mTime].min())
        dict1["TimeMax"] =  str(self.mSignalFrame[self.mTime].max())
        dict1["TimeDelta"] =  str(self.mTimeDelta)
        dict1["Horizon"] =  self.mHorizon;
        return dict1;

    def addVars(self, df):
        df[self.mRowNumberColumn] = self.mSignalFrame[self.mRowNumberColumn]
        df[self.mTime] = self.mSignalFrame[self.mTime]
        df[self.mNormalizedTimeColumn] = self.mSignalFrame[self.mNormalizedTimeColumn]
        df[self.mSignal] = self.mSignalFrame[self.mSignal]
        df[self.mOriginalSignal] = self.mSignalFrame[self.mOriginalSignal]

    def get_time_dtype(self):
        # print(self.mTimeMax, type(self.mTimeMax))
        lType = self.mSignalFrame[self.mTime].dtype;
        return lType;

    def checkDateTypesForNewDataset(self, df):
        if(self.mTimeMax is not None):
            lType1 = self.get_time_dtype();
            lType2 = df[self.mTime].dtype
            if(lType1.kind != lType2.kind):
                raise tsutil.PyAF_Error('Incompatible Time Column Type expected=' + str(lType1) + ' got: ' + str(lType2) + "'");
                pass
        

    def transformDataset(self, df):
        self.checkDateTypesForNewDataset(df);
        # new row
        lLastRow = df.tail(1).copy();
        lNextTime = self.nextTime(df, 1)
        lLastRow[self.mTime] = lNextTime
        lLastRow[self.mSignal] = np.nan
        if(self.mNormalizedTimeColumn in df.columns):
            lLastRow[self.mNormalizedTimeColumn] = self.normalizeTime(lNextTime)
            lLastRow[self.mRowNumberColumn] = lLastRow[self.mRowNumberColumn].max() + 1
        # print(lLastRow.columns ,  df.columns)
        # assert(str(lLastRow.columns) == str(df.columns))

        df = pd.concat([df, lLastRow], ignore_index=True, verify_integrity = True, sort=False); 
        
        if(self.mNormalizedTimeColumn not in df.columns):
            df[self.mRowNumberColumn] = np.arange(0, df.shape[0]);
            df[self.mNormalizedTimeColumn] = self.compute_normalized_date_column(df[self.mTime])
            
        # print(df.tail());
        return df;


    def isPhysicalTime(self):
        lHelper = dtfunc.cDateTime_Helper()
        return lHelper.isPhysicalTime(self.mSignalFrame[self.mTime])

    
    def analyzeSeasonals(self):
        if(not self.isPhysicalTime()):
            return;
        lEstim = self.mSplit.getEstimPart(self.mSignalFrame);
        lEstimTime = lEstim[self.mTime]
        lHelper = dtfunc.cDateTime_Helper()
        self.mResolution = lHelper.guess_time_resolution(lEstimTime);


    def checkDateTypes(self):
        # print(self.mSignalFrame.info());
        type1 = self.mSignalFrame[self.mTime].dtype
        if(type1.kind == 'O'):
            raise tsutil.PyAF_Error('Invalid Time Column Type ' + self.mTime + '[' + str(type1) + ']');
        


    def adaptTimeDeltaToTimeResolution(self):
        if(not self.isPhysicalTime()):
            return;
        lHelper = dtfunc.cDateTime_Helper()
        self.mTimeDelta = lHelper.adaptTimeDeltaToTimeResolution(self.mResolution , self.mTimeDelta);    
    
    def computeTimeDelta(self):
        #print(self.mSignalFrame.columns);
        # print(self.mSignalFrame[self.mTime].head());
        lEstim = self.mSplit.getEstimPart(self.mSignalFrame)
        lTimeBefore = lEstim[self.mTime].shift(1);
        # lTimeBefore.fillna(self.mTimeMin, inplace=True)
        N = lEstim.shape[0];
        if(N == 1):
            if(self.isPhysicalTime()):
                self.mTimeDelta = np.timedelta64(1,'D');
            else:
                self.mTimeDelta = 1
            return
        #print(self.mSignal, self.mTime, N);
        #print(lEstim[self.mTime].head());
        #print(lTimeBefore.head());
        lDiffs = lEstim[self.mTime][1:N] - lTimeBefore[1:N]
        
        if(self.mOptions.mTimeDeltaComputationMethod == "USER"):
            self.mTimeDelta = self.mOptions.mUserTimeDelta;
        if(self.mOptions.mTimeDeltaComputationMethod == "AVG"):
            self.mTimeDelta = np.mean(lDiffs);
            type1 = self.mSignalFrame[self.mTime].dtype
            if(type1.kind == 'i' or type1.kind == 'u'):
                self.mTimeDelta = int(self.mTimeDelta)
        if(self.mOptions.mTimeDeltaComputationMethod == "MODE"):
            delta_counts = pd.DataFrame(lDiffs.value_counts());
            self.mTimeDelta = delta_counts[self.mTime].argmax();
        self.adaptTimeDeltaToTimeResolution();

    def estimate(self):
        #print(self.mSignalFrame.columns);
        #print(self.mSignalFrame[self.mTime].head());
        self.checkDateTypes();
        
        self.mRowNumberColumn = "row_number"
        self.mNormalizedTimeColumn = self.mTime + "_Normalized";

        self.analyzeSeasonals();

        lEstim = self.mSplit.getEstimPart(self.mSignalFrame)
        self.mTimeMin = lEstim[self.mTime].min();
        self.mTimeMax = lEstim[self.mTime].max();
        if(self.isPhysicalTime()):
            self.mTimeMin = np.datetime64(self.mTimeMin.to_pydatetime());
            self.mTimeMax = np.datetime64(self.mTimeMax.to_pydatetime());
        self.mTimeMinMaxDiff = self.mTimeMax - self.mTimeMin;
        self.mEstimCount = lEstim.shape[0]
        # print(self.mTimeMin, self.mTimeMax , self.mTimeMinMaxDiff , (self.mTimeMax - self.mTimeMin)/self.mTimeMinMaxDiff)
        self.computeTimeDelta();
        self.mSignalFrame[self.mNormalizedTimeColumn] = self.compute_normalized_date_column(self.mSignalFrame[self.mTime])
        self.dump();

    def dump(self):
        time_info = self.info(); 
        

    def compute_normalized_date_column(self, idate_column):
        if(self.mEstimCount == 1):
            return 0.0;
        return self.normalizeTime(idate_column)

    def normalizeTime(self , iTime):
        if(self.mEstimCount == 1):
            return 0.0;
        output =  ( iTime- self.mTimeMin) / self.mTimeMinMaxDiff
        return output


    def cast_to_time_dtype(self, iTimeValue):
        lType1 = self.get_time_dtype();
        lTimeValue = np.array([iTimeValue]).astype(lType1)[0];
        return lTimeValue;
    
    def nextTime(self, df, iSteps):
        #print(df.tail(1)[self.mTime]);
        lLastTime = df[self.mTime].values[-1]
        if(self.isPhysicalTime()):
            lLastTime = pd.Timestamp(lLastTime)
            # print("NEXT_TIME" , lLastTime, iSteps, self.mTimeDelta);
            lNextTime = lLastTime + iSteps * self.mTimeDelta;
            lNextTime = self.cast_to_time_dtype(lNextTime.to_datetime64())
        else:
            lNextTime = lLastTime + iSteps * self.mTimeDelta;
            lNextTime = self.cast_to_time_dtype(lNextTime)
            
            
        return lNextTime;
