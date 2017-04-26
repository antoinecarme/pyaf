# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np
import datetime as dt
import calendar

from . import Utils as tsutil


from dateutil.relativedelta import relativedelta

class cTimeInfo:

    def __init__(self):
        self.mSignalFrame = pd.DataFrame()
        self.mTimeMin = None;
        self.mTimeMax = None;
        self.mTimeMinMaxDiff = None;
        self.mTimeDelta = None;
        self.mHorizon = None;        
        self.RES_NONE = 0
        self.RES_SECOND = 1
        self.RES_MINUTE = 2
        self.RES_HOUR = 3
        self.RES_DAY = 4
        self.RES_MONTH = 5
        self.mResolution = self.RES_NONE
        self.mSecondsInResolution = {};
        self.mSecondsInResolution[self.RES_NONE] = 0;
        self.mSecondsInResolution[self.RES_SECOND] = 1;
        self.mSecondsInResolution[self.RES_MINUTE] = 1 * 60;
        self.mSecondsInResolution[self.RES_HOUR] = 1 * 60 * 60;
        self.mSecondsInResolution[self.RES_DAY] = 1 * 60 * 60 * 24;
        self.mSecondsInResolution[self.RES_MONTH] = 1 * 60 * 60 * 24 * 30;
        

    def info(self):
        lStr2 = "TimeVariable='" + self.mTime +"'";
        lStr2 += " TimeMin=" + str(self.mTimeMin) +"";
        lStr2 += " TimeMax=" + str(self.mTimeMax) +"";
        lStr2 += " TimeDelta=" + str(self.mTimeDelta) +"";
        lStr2 += " Estimation = (" + str(self.mEstimStart) + " , " + str(self.mEstimEnd) + ")";
        lStr2 += " Validation = (" + str(self.mValidStart) + " , " + str(self.mValidEnd) + ")";
        lStr2 += " Test = (" + str(self.mTestStart) + " , " + str(self.mTestEnd) + ")";
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
        series = pd.Series([self.mTimeMax]);
        lType = np.dtype(series);
        return lType;

    def cast_to_time_dtype(self, iTimeValue):
        lType1 = self.get_time_dtype();
        lTimeValue = np.array([iTimeValue]).astype(lType1)[0];
        return lTimeValue;

    def checkDateAndSignalTypesForNewDataset(self, df):
        if(self.mTimeMax is not None):
            lType1 = self.get_time_dtype();
            lType2 = np.dtype(df[self.mTime]);
            if(lType1 != lType2):
                raise tsutil.ForecastError('Incompatible Time Column Type expected=' + str(lType1) + ' got: ' + str(lType2) + "'");
        

    def transformDataset(self, df):
        self.checkDateAndSignalTypesForNewDataset(df);
        # new row
        lLastRow = df.tail(1).copy();
        lLastRow[self.mTime] = self.nextTime(df, 1);
        lLastRow[self.mSignal] = np.nan;
        df = df.append(lLastRow, ignore_index=True, verify_integrity = True);        
        df[self.mRowNumberColumn] = np.arange(0, df.shape[0]);
        df[self.mNormalizedTimeColumn] = self.normalizeTime(df[self.mTime]);
        # print(df.tail());
        return df;


    def isPhysicalTime(self):
        type1 = np.dtype(self.mSignalFrame[self.mTime])
        return (type1.kind == 'M');


    def get_date_part_value(self , iTime, iDatePart):
        if(iDatePart == "Year"):
            return iTime.year;
        if(iDatePart == "MonthOfYear"):
            return iTime.month;
        if(iDatePart == "DayOfMonth"):
            return iTime.day;
        if(iDatePart == "Hour"):
            return iTime.hour;
        if(iDatePart == "Min"):
            return iTime.minute;
        if(iDatePart == "Second"):
            return iTime.second;
        # TODO : this one is not very easy ... too many values.
        # need long time series to have reliable signal means 
        #        if(self.mDatePart == "DayOfYear"):
        #            return iTime.dayofyear;
        if(iDatePart == "WeekOfYear"):
            return iTime.weekofyear;
        if(iDatePart == "DayOfWeek"):
            return iTime.dayofweek;
        return 0.0;
    
    def analyzeSeasonals(self):
        if(not self.isPhysicalTime()):
            return;
        lEstim = self.getEstimPart(self.mSignalFrame);
        lEstimSecond = lEstim[self.mTime].apply(
            lambda x : self.get_date_part_value(x , "Second"));
        if(lEstimSecond.nunique() > 1.0):
            self.mResolution = self.RES_SECOND;
            return;
        lEstimMinute = lEstim[self.mTime].apply(        
            lambda x : self.get_date_part_value(x , "Minute"));
        if(lEstimMinute.nunique() > 1.0):
            self.mResolution =  self.RES_MINUTE;
            return;
        lEstimHour = lEstim[self.mTime].apply(        
            lambda x : self.get_date_part_value(x , "Hour"));
        if(lEstimHour.nunique() > 1.0):
            self.mResolution =  self.RES_HOUR;
            return;
        lEstimDayOfMonth = lEstim[self.mTime].apply(        
            lambda x : self.get_date_part_value(x , "DayOfMonth"));
        if(lEstimDayOfMonth.nunique() > 1.0):
            self.mResolution =  self.RES_DAY;
            return;
        lEstimMonth = lEstim[self.mTime].apply(       
            lambda x : self.get_date_part_value(x , "MonthOfYear"));
        if(lEstimMonth.nunique() > 1.0):
            self.mResolution =  self.RES_MONTH;
            return;

    def getSecondsInResolution(self):
        return self.mSecondsInResolution.get(self.mResolution , 0.0);

    def cutFrame(self, df):
        lFrameFit = df[self.mEstimStart : self.mEstimEnd];
        lFrameForecast = df[self.mValidStart : self.mValidEnd];
        lFrameTest = df[self.mTestStart : self.mTestEnd];
        return (lFrameFit, lFrameForecast, lFrameTest)

    def getEstimPart(self, df):
        lFrameFit = df[self.mEstimStart : self.mEstimEnd];
        return lFrameFit;

    def getValidPart(self, df):
        lFrameValid = df[self.mValidStart : self.mValidEnd];
        return lFrameValid;

    def defineCuttingParameters(self):
        lStr = "CUTTING_START SignalVariable='" + self.mSignal +"'";
        # print(lStr);
        #print(self.mSignalFrame.head())
        self.mTrainSize = self.mSignalFrame.shape[0];
        assert(self.mTrainSize > 0);
        lEstEnd = int((self.mTrainSize - self.mHorizon) * self.mOptions.mEstimRatio);
        lValSize = self.mTrainSize - self.mHorizon - lEstEnd;
        lTooSmall = False;
        # training too small
        if((self.mTrainSize < 30) or (lValSize < self.mHorizon)):
            lTooSmall = True;
        
        if(lTooSmall):
            self.mEstimStart = 0;
            self.mEstimEnd = self.mTrainSize;
            self.mValidStart = 0;
            self.mValidEnd = self.mTrainSize;
            self.mTestStart = 0;
            self.mTestEnd = self.mTrainSize;
        else:
            self.mEstimStart = 0;
            self.mEstimEnd = lEstEnd;
            self.mValidStart = self.mEstimEnd;
            self.mValidEnd = self.mTrainSize - self.mHorizon;
            self.mTestStart = self.mValidEnd;
            self.mTestEnd = self.mTrainSize;

        lStr = "CUTTING_PARAMETERS " + str(self.mTrainSize) + " Estimation = (" + str(self.mEstimStart) + " , " + str(self.mEstimEnd) + ")";
        lStr += " Validation = (" + str(self.mValidStart) + " , " + str(self.mValidEnd) + ")";
        lStr += " Test = (" + str(self.mTestStart) + " , " + str(self.mTestEnd) + ")";
        #print(lStr);
        
        pass

    def checkDateAndSignalTypes(self):
        # print(self.mSignalFrame.info());
        type1 = np.dtype(self.mSignalFrame[self.mTime])
        if(type1.kind == 'O'):
            raise tsutil.ForecastError('Invalid Time Column Type ' + self.mTime + '[' + str(type1) + ']');
        type2 = np.dtype(self.mSignalFrame[self.mSignal])
        if(type2.kind == 'O'):
            raise tsutil.ForecastError('Invalid Signal Column Type ' + self.mSignal);
        

    def round_datetime_to_seconds(self, iDate):
        lDate = dt.datetime.utcfromtimestamp(iDate.astype(int) * 1e-9)
        lDate0 = dt.datetime(lDate.year, 1 , 1, 0, 0 , 0) 
        delta1 = (lDate - lDate0)
        rounded_sec = round(delta1.total_seconds())
        delta_sec = dt.timedelta(seconds=rounded_sec)
        lDate1 = lDate0 + delta_sec
        # print(iDate.isoformat() , "\t" , lDate0.isoformat(), "\t", lDate1.isoformat())
        return lDate1;

    def isOneRowDataset(self):
        return ((1 + self.mEstimStart) ==  self.mEstimEnd)

    def computeTimeDelta(self):
        #print(self.mSignalFrame.columns);
        # print(self.mSignalFrame[self.mTime].head());
        lEstim = self.mSignalFrame[self.mEstimStart : self.mEstimEnd]
        lTimeBefore = lEstim[self.mTime].shift(1);
        # lTimeBefore.fillna(self.mTimeMin, inplace=True)
        N = lEstim.shape[0];
        if(N == 1):
            if(self.isPhysicalTime()):
                self.mTimeDelta = pd.Timedelta(seconds=1);
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
        if(self.mOptions.mTimeDeltaComputationMethod == "MODE"):
            delta_counts = pd.DataFrame(lDiffs.value_counts());
            self.mTimeDelta = delta_counts[self.mTime].argmax();
        if(self.isPhysicalTime()):
            rounded_sec = round(self.mTimeDelta.total_seconds());
            self.mTimeDelta = pd.Timedelta(seconds=rounded_sec);
            # print(type(self.mTimeDelta), self.mTimeDelta)

    def estimate(self):
        #print(self.mSignalFrame.columns);
        #print(self.mSignalFrame[self.mTime].head());
        self.checkDateAndSignalTypes();
        
        self.mRowNumberColumn = "row_number"
        self.mNormalizedTimeColumn = self.mTime + "_Normalized";

        self.defineCuttingParameters();

        self.analyzeSeasonals();

        self.mSecondsInResolution = self.getSecondsInResolution();
        lEstim = self.mSignalFrame[self.mEstimStart : self.mEstimEnd]
        self.mTimeMin = lEstim[self.mTime].min();
        self.mTimeMax = lEstim[self.mTime].max();
        self.mTimeMinMaxDiff = self.mTimeMax - self.mTimeMin;
        
        self.computeTimeDelta();
        self.mSignalFrame[self.mNormalizedTimeColumn] = self.normalizeTime(self.mSignalFrame[self.mTime])
        self.dump();

    def dump(self):
        time_info = self.info(); 
        

    def normalizeTime(self , iTime):
        if(self.isOneRowDataset()):
            return 0.0;
        return (iTime - self.mTimeMin) / self.mTimeMinMaxDiff

    def addMonths(self, iTime , iMonths):    
        lTime = dt.datetime.utcfromtimestamp(iTime.astype(int) * 1e-9)
        date_after_month = lTime + relativedelta(months=iMonths)
        #print(lTime, iMonths, date_after_month);
        lDate = np.datetime64(date_after_month)
        return lDate;
    
    def nextTime(self, df, iSteps):
        #print(df.tail(1)[self.mTime]);
        lLastTime = df[self.mTime].values[-1]
        # Better handle time delta in months
        # print("NEXT_TIME" , lLastTime, iSteps, self.mTimeDelta);
        lNextTime = lLastTime + iSteps * self.mTimeDelta;
        if(self.mResolution == self.RES_MONTH):
            lMonths = int(iSteps * self.mTimeDelta.days / 30.0);
            lNextTime = self.addMonths(lLastTime, lMonths);
        if(self.mOptions.mBusinessDaysOnly):
            lOffset = [1, 1, 1, 1, 3, 2, 1][lNextTime.weekday()];
            lNextTime = lNextTime + dt.timedelta(days = lOffset);

        lNextTime = self.cast_to_time_dtype(lNextTime);        
        if(self.isPhysicalTime()):
            lNextTime = self.round_datetime_to_seconds(lNextTime);
        return lNextTime;
