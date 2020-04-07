# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np
from enum import IntEnum

from . import Utils as tsutil
from . import TimeSeries_Cutting as tscut

class eDatePart(IntEnum):
    Second = 1
    Minute = 2
    Hour = 3
    DayOfWeek = 4
    DayOfMonth = 5
    MonthOfYear = 6
    WeekOfYear = 7
    DayOfYear = 8

class eTimeResolution(IntEnum):
    NONE = 0
    SECOND = 1
    MINUTE = 2
    HOUR = 3
    DAY = 4
    MONTH = 5
    YEAR = 6
    
    
class cTimeInfo:
    # class data

    def __init__(self):
        self.mSignalFrame = pd.DataFrame()
        self.mTimeMin = None;
        self.mTimeMax = None;
        self.mTimeMinMaxDiff = None;
        self.mTimeDelta = None;
        self.mHorizon = None;        
        self.mResolution = eTimeResolution.NONE
        self.mSplit = None

    def info(self):
        lStr2 = "TimeVariable='" + self.mTime +"'";
        lStr2 += " TimeMin=" + str(self.mTimeMin) +"";
        lStr2 += " TimeMax=" + str(self.mTimeMax) +"";
        lStr2 += " TimeDelta=" + str(self.mTimeDelta) +"";
        lStr2 += " Horizon=" + str(self.mHorizon) +"";
        return lStr2;


    def to_json(self):
        dict1 = {};
        dict1["TimeVariable"] =  self.mTime;
        dict1["TimeMinMax"] =  [str(self.mSignalFrame[self.mTime].min()) ,
                                str(self.mSignalFrame[self.mTime].max())];
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
        lType = np.dtype(self.mTimeMax);
        return lType;

    def cast_to_time_dtype(self, iTimeValue):
        lType1 = self.get_time_dtype();
        lTimeValue = np.array([iTimeValue]).astype(lType1)[0];
        return lTimeValue;

    def checkDateAndSignalTypesForNewDataset(self, df):
        if(self.mTimeMax is not None):
            lType1 = self.get_time_dtype();
            lType2 = np.dtype(df[self.mTime]);
            if(lType1.kind != lType2.kind):
                raise tsutil.PyAF_Error('Incompatible Time Column Type expected=' + str(lType1) + ' got: ' + str(lType2) + "'");
                pass
        

    def transformDataset(self, df):
        self.checkDateAndSignalTypesForNewDataset(df);
        # new row
        lLastRow = df.tail(1).copy();
        lNextTime = self.nextTime(df, 1)
        lLastRow[self.mTime] = lNextTime
        lLastRow[self.mSignal] = np.nan
        if(self.mNormalizedTimeColumn in df.columns):
            lLastRow[self.mNormalizedTimeColumn] = self.normalizeTime(lNextTime)
            lLastRow[self.mRowNumberColumn] = lLastRow[self.mRowNumberColumn].max() + 1
        # print(lLastRow.columns ,  df.columns)
        assert(str(lLastRow.columns) == str(df.columns))
        df = df.append(lLastRow, ignore_index=True, verify_integrity = True, sort=False);        
        if(self.mNormalizedTimeColumn not in df.columns):
            df[self.mRowNumberColumn] = np.arange(0, df.shape[0]);
            df[self.mNormalizedTimeColumn] = self.compute_normalized_date_column(df[self.mTime])
            
        # print(df.tail());
        return df;


    def isPhysicalTime(self):
        type1 = np.dtype(self.mSignalFrame[self.mTime])
        return (type1.kind == 'M');


    def apply_date_time_computer(self, iDatePart, series):
        lOut = None
        if(iDatePart == eDatePart.Second):
            lOut = series.dt.second
        elif(iDatePart == eDatePart.Minute):
            lOut = series.dt.minute
        elif(iDatePart == eDatePart.Hour):
            lOut = series.dt.hour
        elif(iDatePart == eDatePart.DayOfWeek):
            lOut = series.dt.dayofweek
        elif(iDatePart == eDatePart.DayOfMonth):
            lOut = series.dt.day
        elif(iDatePart == eDatePart.DayOfYear):
            lOut = series.dt.dayofyear
        elif(iDatePart == eDatePart.MonthOfYear):
            lOut = series.dt.month
        elif(iDatePart == eDatePart.WeekOfYear):
            lOut = series.dt.week
        if(lOut is None):
            print("apply_date_time_computer_failures" , iDatePart)
        assert(lOut is not None)
        return lOut
    
    def analyzeSeasonals(self):
        if(not self.isPhysicalTime()):
            return;
        lEstim = self.mSplit.getEstimPart(self.mSignalFrame);
        lEstimTime = lEstim[self.mTime]
        lEstimSecond = self.apply_date_time_computer(eDatePart.Second, lEstimTime)
        if(lEstimSecond.nunique() > 1.0):
            self.mResolution = eTimeResolution.SECOND;
            return;
        lEstimMinute = self.apply_date_time_computer(eDatePart.Minute, lEstimTime)
        if(lEstimMinute.nunique() > 1.0):
            self.mResolution =  eTimeResolution.MINUTE;
            return;
        lEstimHour = self.apply_date_time_computer(eDatePart.Hour, lEstimTime)
        if(lEstimHour.nunique() > 1.0):
            self.mResolution =  eTimeResolution.HOUR;
            return;
        lEstimDayOfMonth = self.apply_date_time_computer(eDatePart.DayOfMonth, lEstimTime)
        if(lEstimDayOfMonth.nunique() > 1.0):
            self.mResolution =  eTimeResolution.DAY;
            return;
        lEstimMonth = self.apply_date_time_computer(eDatePart.MonthOfYear, lEstimTime)
        if(lEstimMonth.nunique() > 1.0):
            self.mResolution =  eTimeResolution.MONTH;
            return;
        self.mResolution =  eTimeResolution.YEAR;


    def checkDateAndSignalTypes(self):
        # print(self.mSignalFrame.info());
        type1 = np.dtype(self.mSignalFrame[self.mTime])
        if(type1.kind == 'O'):
            raise tsutil.PyAF_Error('Invalid Time Column Type ' + self.mTime + '[' + str(type1) + ']');
        type2 = np.dtype(self.mSignalFrame[self.mSignal])
        if(type2.kind == 'O'):
            raise tsutil.PyAF_Error('Invalid Signal Column Type ' + self.mSignal);
        


    def adaptTimeDeltaToTimeResolution(self):
        if(not self.isPhysicalTime()):
            return;
        if(eTimeResolution.SECOND == self.mResolution):
            self.mTimeDelta = pd.DateOffset(seconds=round(self.mTimeDelta / np.timedelta64(1,'s')))
            return;
        if(eTimeResolution.MINUTE == self.mResolution):
            self.mTimeDelta = pd.DateOffset(minutes=round(self.mTimeDelta / np.timedelta64(1,'m')))
            return;
        if(eTimeResolution.HOUR == self.mResolution):
            self.mTimeDelta = pd.DateOffset(hours=round(self.mTimeDelta / np.timedelta64(1,'h')))
            return;
        if(eTimeResolution.DAY == self.mResolution):
            self.mTimeDelta = pd.DateOffset(days=round(self.mTimeDelta / np.timedelta64(1,'D')))
            return;
        if(eTimeResolution.MONTH == self.mResolution):
            self.mTimeDelta = pd.DateOffset(months=round(self.mTimeDelta // np.timedelta64(30,'D')))
            return;
        if(eTimeResolution.YEAR == self.mResolution):
            self.mTimeDelta = pd.DateOffset(months=round(self.mTimeDelta // np.timedelta64(365,'D')))
            return;
        pass
    
    def get_lags_for_time_resolution(self):
        if(not self.isPhysicalTime()):
            return None;
        lARORder = {}
        lARORder[eTimeResolution.SECOND] = 60
        lARORder[eTimeResolution.MINUTE] = 60
        lARORder[eTimeResolution.HOUR] = 24
        lARORder[eTimeResolution.DAY] = 31
        lARORder[eTimeResolution.MONTH] = 12
        return lARORder.get(self.mResolution , None)
    
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
            type1 = np.dtype(self.mSignalFrame[self.mTime])
            if(type1.kind == 'i' or type1.kind == 'u'):
                self.mTimeDelta = int(self.mTimeDelta)
        if(self.mOptions.mTimeDeltaComputationMethod == "MODE"):
            delta_counts = pd.DataFrame(lDiffs.value_counts());
            self.mTimeDelta = delta_counts[self.mTime].argmax();
        self.adaptTimeDeltaToTimeResolution();

    def estimate(self):
        #print(self.mSignalFrame.columns);
        #print(self.mSignalFrame[self.mTime].head());
        self.checkDateAndSignalTypes();
        
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
        return idate_column.apply(self.normalizeTime)

    @tsutil.cMemoize
    def normalizeTime(self , iTime):
        if(self.mEstimCount == 1):
            return 0.0;
        output =  ( iTime- self.mTimeMin) / self.mTimeMinMaxDiff
        return output

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
