# Copyright (C) 2016 Antoine Carme <Antoine.Carme@outlook.com>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np
from enum import IntEnum

from . import Utils as tsutil

class eDatePart(IntEnum):
    Second = 1
    Minute = 2
    Hour = 3
    DayOfWeek = 4
    DayOfMonth = 5
    MonthOfYear = 6
    WeekOfYear = 7
    DayOfYear = 8
    HourOfWeek = 9
    TwoHourOfWeek = 10
    ThreeHourOfWeek = 11
    FourHourOfWeek = 12
    SixHourOfWeek = 13
    EightHourOfWeek = 14
    TwelveHourOfWeek = 15
    WeekOfMonth = 16
    DayOfNthWeekOfMonth = 17

class eTimeResolution(IntEnum):
    NONE = 0
    SECOND = 1
    MINUTE = 2
    HOUR = 3
    DAY = 4
    MONTH = 5
    YEAR = 6
    
    
class cDateTime_Helper:

    def get_week_of_year(series):
        return series.dt.isocalendar().week
    
    def get_week_of_month(series):
        lFirstDayOfMonth = series - pd.to_timedelta(series.dt.day - 1, unit='D')
        return series.dt.isocalendar().week - lFirstDayOfMonth.dt.isocalendar().week + 1



    gComputers = {
        eDatePart.Second: lambda series : series.dt.second,
        eDatePart.Minute: lambda series : series.dt.minute,
        eDatePart.Hour:   lambda series : series.dt.hour,
        eDatePart.DayOfWeek: lambda series : series.dt.dayofweek,
        eDatePart.HourOfWeek: lambda series : series.dt.dayofweek * 24 + series.dt.hour,
        eDatePart.TwoHourOfWeek: lambda series : series.dt.dayofweek * 12 + series.dt.hour // 2,
        eDatePart.ThreeHourOfWeek: lambda series : series.dt.dayofweek * 8 + series.dt.hour // 3,
        eDatePart.FourHourOfWeek: lambda series : series.dt.dayofweek * 6 + series.dt.hour // 4,
        eDatePart.SixHourOfWeek: lambda series : series.dt.dayofweek * 4 + series.dt.hour // 6,
        eDatePart.EightHourOfWeek: lambda series : series.dt.dayofweek * 3 + series.dt.hour // 8,
        eDatePart.TwelveHourOfWeek: lambda series : series.dt.dayofweek * 2 + series.dt.hour // 12,
        eDatePart.DayOfMonth: lambda series : series.dt.day,
        eDatePart.DayOfYear: lambda series : series.dt.dayofyear,
        eDatePart.MonthOfYear: lambda series : series.dt.month,
        eDatePart.WeekOfYear:  lambda series : cDateTime_Helper.get_week_of_year(series),
        eDatePart.WeekOfMonth: get_week_of_month,
        eDatePart.DayOfNthWeekOfMonth: lambda series : cDateTime_Helper.get_week_of_month(series) * 7 + series.dt.dayofweek
    }

    def get_date_time_computer(iDatePart):
        lComputer = cDateTime_Helper.gComputers.get(iDatePart)
        return lComputer    
    
    def __init__(self):
        pass

    def apply_date_time_computer(self, iDatePart, series):
        # Future Warning regarding DateTime_Functions - series.dt.weekofyear #153
        lComputer = cDateTime_Helper.get_date_time_computer(iDatePart)
        if(lComputer is None):
            print("apply_date_time_computer_failures" , iDatePart)
        assert(lComputer is not None)
        lOut = lComputer(series)
        return lOut

    def guess_time_resolution(self, iEstimTime):
        lEstimSecond = self.apply_date_time_computer(eDatePart.Second, iEstimTime)
        if(lEstimSecond.nunique() > 1.0):
            return eTimeResolution.SECOND;
        lEstimMinute = self.apply_date_time_computer(eDatePart.Minute, iEstimTime)
        if(lEstimMinute.nunique() > 1.0):
            return eTimeResolution.MINUTE;
        lEstimHour = self.apply_date_time_computer(eDatePart.Hour, iEstimTime)
        if(lEstimHour.nunique() > 1.0):
            return eTimeResolution.HOUR;
        lEstimDayOfMonth = self.apply_date_time_computer(eDatePart.DayOfMonth, iEstimTime)
        if(lEstimDayOfMonth.nunique() > 1.0):
            return eTimeResolution.DAY;
        lEstimMonth = self.apply_date_time_computer(eDatePart.MonthOfYear, iEstimTime)
        if(lEstimMonth.nunique() > 1.0):
            return eTimeResolution.MONTH;
        # fallback
        return eTimeResolution.YEAR;

    
    def get_lags_for_time_resolution(self):
        # self.mOptions.mMaxAROrder is set to 64 by default, Which covers all these resolutions.
        if(not self.isPhysicalTime()):
            return None;
        lARORder = {}
        lARORder[eTimeResolution.SECOND] = 60
        lARORder[eTimeResolution.MINUTE] = 60
        lARORder[eTimeResolution.HOUR] = 24
        lARORder[eTimeResolution.DAY] = 31
        lARORder[eTimeResolution.MONTH] = 12
        return lARORder.get(self.mResolution , None)


    def adaptTimeDeltaToTimeResolution(self, iResolution, iTimeDelta):
        if(eTimeResolution.SECOND == iResolution):
            lTimeDelta = pd.DateOffset(seconds=round(iTimeDelta / np.timedelta64(1,'s')))
            return lTimeDelta
        if(eTimeResolution.MINUTE == iResolution):
            lTimeDelta = pd.DateOffset(minutes=round(iTimeDelta / np.timedelta64(1,'m')))
            return lTimeDelta
        if(eTimeResolution.HOUR == iResolution):
            lTimeDelta = pd.DateOffset(hours=round(iTimeDelta / np.timedelta64(1,'h')))
            return lTimeDelta
        if(eTimeResolution.DAY == iResolution):
            lTimeDelta = pd.DateOffset(days=round(iTimeDelta / np.timedelta64(1,'D')))
            return lTimeDelta
        if(eTimeResolution.MONTH == iResolution):
            lTimeDelta = pd.DateOffset(months=round(iTimeDelta // np.timedelta64(30,'D')))
            return lTimeDelta
        if(eTimeResolution.YEAR == iResolution):
            lTimeDelta = pd.DateOffset(months=round(iTimeDelta // np.timedelta64(365,'D')))
            return lTimeDelta
        pass
    
    
    def get_beginning_of_period(self, iPeriod, x):
        # iPeriod can be "H" (hour), "D" (day), etc
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        return x.to_period(iPeriod).start_time

    
    def computeTimeFrequency_in_seconds(self, iTime):
        lTimeBefore = iTime.shift(1);
        lDiffs = iTime[1:] - lTimeBefore[1:]
        lDelta = np.min(lDiffs);
        return lDelta.total_seconds()

    def get_period_length_in_seconds(self, iFreq):
        # keep this test as  simple as possible. Use higher level pandas API.
        # Avoid digging into pandas low-level time sampling details (total seconds in small time range of two periods).
        # sampel output : '1W' => 604800
        lRange = pd.date_range('1970-01-01', periods=2, freq=iFreq)
        delta_t = lRange[1] - lRange[0]
        return delta_t.total_seconds()


    def isPhysicalTime(self, iTimeColumn):
        type1 = iTimeColumn.dtype
        return (type1.kind == 'M');
        
        
