# Copyright (C) 2016 Antoine Carme <Antoine.Carme@outlook.com>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np
from enum import IntEnum

from . import Utils as tsutil

class eDatasetType(IntEnum):
    Fit = 1
    Forecast = 2
    Test = 3


class cCuttingInfo:
    def __init__(self):
        pass


    def estimate(self):
        self.defineCuttingParameters();

    def set_default_split(self):
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

    def check_split(self, iSplit):
        if(len(iSplit) != 3):
            raise tsutil.PyAF_Error('Invalid Split ' + str(iSplit));
        if(iSplit[0] < 0.0 or iSplit[0] > 1.0):
            raise tsutil.PyAF_Error('Invalid Estimation Ratio ' + str(iSplit[0]));
        if(iSplit[1] < 0.0 or iSplit[1] > 1.0):
            raise tsutil.PyAF_Error('Invalid Validation Ratio ' + str(iSplit[1]));
        if(iSplit[2] < 0.0 or iSplit[2] > 1.0):
            raise tsutil.PyAF_Error('Invalid Test Ratio ' + str(iSplit[2]));
        lTotal =  iSplit[0] + iSplit[1] + iSplit[2]
        if(lTotal < 0 or lTotal > 1):
            raise tsutil.PyAF_Error('Invalid Split Ratio Sum' + str(iSplit));

            
    def set_split(self, iSplit):
        self.mTrainSize = self.mSignalFrame.shape[0];
        assert(self.mTrainSize > 0);
        self.check_split(iSplit)
        lEstEnd = int(self.mTrainSize * iSplit[0]);
        lValSize = int(self.mTrainSize * iSplit[1]);
        lTestSize = int(self.mTrainSize * iSplit[2]);
        
        self.mEstimStart = 0;
        self.mEstimEnd = lEstEnd;
        self.mValidStart = self.mEstimEnd;
        self.mValidEnd = self.mValidStart + lValSize;
        self.mTestStart = self.mValidEnd;
        self.mTestEnd = self.mTestStart + lTestSize;
        
    def defineCuttingParameters(self):
        lStr = "CUTTING_START SignalVariable='" + self.mSignal +"'";
        # tsutil.print_pyaf_detailed_info(lStr);
        # tsutil.print_pyaf_detailed_info(self.mSignalFrame.head())
        if(self.mOptions.mCustomSplit is not None):
            self.set_split(self.mOptions.mCustomSplit)
        else:
            self.set_default_split()

        lStr = "CUTTING_PARAMETERS " + str(self.mTrainSize) + " Estimation = (" + str(self.mEstimStart) + " , " + str(self.mEstimEnd) + ")";
        lStr += " Validation = (" + str(self.mValidStart) + " , " + str(self.mValidEnd) + ")";
        lStr += " Test = (" + str(self.mTestStart) + " , " + str(self.mTestEnd) + ")";
        # tsutil.print_pyaf_detailed_info(lStr);
        
        pass

    def cutFrame(self, df):
        lDict = {
            eDatasetType.Fit : df[self.mEstimStart : self.mEstimEnd],
            eDatasetType.Forecast : df[self.mValidStart : self.mValidEnd],
            eDatasetType.Test : df[self.mTestStart : self.mTestEnd]
        }
        return lDict

    def getEstimPart(self, df):
        lFrameFit = df[self.mEstimStart : self.mEstimEnd];
        return lFrameFit;

    def getValidPart(self, df):
        lFrameValid = df[self.mValidStart : self.mValidEnd];
        return lFrameValid;

    def add_dataset_indicators(self, df):
        lCutting = self.cutFrame(df)
        for (lDataset, lFrame) in lCutting.items():
            lColumnName = "dataset_indicator_" + lDataset.name
            df[lColumnName] = np.nan
            df.loc[lFrame.index, lColumnName] = 1.0
        return df

    def info(self):
        lStr2 += " Estimation = (" + str(self.mEstimStart) + " , " + str(self.mEstimEnd) + ")";
        lStr2 += " Validation = (" + str(self.mValidStart) + " , " + str(self.mValidEnd) + ")";
        lStr2 += " Test = (" + str(self.mTestStart) + " , " + str(self.mTestEnd) + ")";
        lStr2 += " Horizon=" + str(self.mHorizon) +"";
        return lStr2;

