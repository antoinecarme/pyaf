# Copyright (C) 2016 Antoine Carme <Antoine.Carme@outlook.com>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license



import pandas as pd
import numpy as np


from . import Utils as tsutil

class cExogenousInfo:

    def __init__(self):
        self.mExogenousVariables = None;
        self.mEncodedExogenous = None;
        self.mExogenousVariableCategories = None;
        self.mExogenousDataFrame = None;
        self.mExogenousData = None;        
        self.mDateVariable = None;
        self.mContExogenousStats = None;
        self.mExcluded = [];
        
    def info(self):
        lStr2 = "ExogenousVariables = '" + self.mExogD +"'";
        return lStr2;


    def to_dict(self):
        dict1 = {};
        return dict1;

    def check(self):
        if(self.mExogenousData is not None):
            lExogenousDataFrame = self.mExogenousData[0];
            lExogenousVariables = self.mExogenousData[1];
            if(self.mDateVariable not in lExogenousDataFrame.columns):
                raise tsutil.PyAF_Error("PYAF_ERROR_TIME_COLUMN_NOT_FOUND_IN_EXOGENOUS " + str(self.mDateVariable));
            for exog in lExogenousVariables:
                if(exog not in lExogenousDataFrame.columns):
                    raise tsutil.PyAF_Error("PYAF_ERROR_EXOGENOUS_VARIABLE_NOT_FOUND " + str(exog));
                    
    def fit(self):
        self.mExogenousDataFrame = self.mExogenousData[0];
        self.mExogenousVariables = self.mExogenousData[1];
        self.mDateVariable = self.mTimeInfo.mTime;
        self.check()
        # print("preProcessExogenousVariables , columns", self.mExogenousVariables);
        self.updateExogenousVariableInfo();
        self.createEncodedExogenous();
        # print("preProcessExogenousVariables , dummy columns", self.mEncodedExogenous);

    def addVars(self, df):
        lExogDate = self.mDateVariable;
        N = df.shape[0]
        df1 = df[[self.mDateVariable]];
        lCompleteEncoded = df1.merge(self.mEncodedExogenousDataFrame,
                                     how='left',
                                     left_on=self.mDateVariable,
                                     right_on=lExogDate);
        lCompleteEncoded.fillna(0.0, inplace=True);
        assert(lCompleteEncoded.shape[0] == N)
        
        df2 = df.merge(lCompleteEncoded,
                       how='left',
                       left_on=self.mDateVariable,
                       right_on=lExogDate);
        return df2;

    def transformDataset(self, df):
        df1 = self.addVars(df);
        return df1;
        
    def createEncodedExogenous(self):
        self.mEncodedExogenous = [];
        lEncodedVars = {self.mDateVariable : self.mExogenousDataFrame[self.mDateVariable]}
        for exog in self.mExogenousVariables:
            if(exog not in self.mExcluded):
                lSeries = self.mExogenousDataFrame[exog]
                lList = self.mExogenousVariableCategories[exog];
                if(lList is not None):
                    for lCat in lList:
                        lDummyName = exog + "=" + str(lCat);
                        lEncodedVars[lDummyName] = np.where(lSeries == lCat , np.int8(1), np.int8(0));
                        self.mEncodedExogenous = self.mEncodedExogenous + [lDummyName];
                else:
                    lExogStats = self.mContExogenousStats[exog];
                    # single precision here ...
                    lStandardized = (lSeries - lExogStats[0])/ lExogStats[1];
                    lStandardized = lStandardized.astype(np.float32);
                    lStandardized.fillna(np.float32(0.0), inplace=True);
                    lEncodedVars[exog] = lStandardized
                    self.mEncodedExogenous = self.mEncodedExogenous + [exog];
            else:
                # print("EXCLUDED" , exog);
                pass
        self.mEncodedExogenousDataFrame = pd.DataFrame(lEncodedVars,
                                                       index = self.mExogenousDataFrame.index);
        # self.mEncodedExogenousDataFrame.info()

    def updateExogenousVariableInfo(self):
        # self.mExogenousDataFrame.info()
        # print(self.mExogenousDataFrame.describe());
        self.mExogenousVariableCategories = {};
        self.mContExogenousStats = {};
        # Compute these stats only on the estimation part.
        (lTimeMin , lTimeMax) = (self.mTimeInfo.mTimeMin , self.mTimeInfo.mTimeMax)
        if(self.mTimeInfo.isPhysicalTime()):
            (lTimeMin , lTimeMax) = (pd.Timestamp(self.mTimeInfo.mTimeMin) , pd.Timestamp(self.mTimeInfo.mTimeMax))
            
        lEstimFrame = self.mExogenousDataFrame[self.mExogenousDataFrame[self.mDateVariable] >= lTimeMin]
        lEstimFrame = lEstimFrame[lEstimFrame[self.mDateVariable] <= lTimeMax]
        for exog in self.mExogenousVariables:
            lType = self.mExogenousDataFrame[exog].dtype
            # print("EXOG_DTYPE" , exog, lType)
            if((lType == object) or (lType.name == 'category')):
                # use nan as a category
                lVC = lEstimFrame[exog].value_counts(dropna = False, sort=False);
                lVC = lVC[lVC > 5]
                lVC = lVC.reset_index().sort_values(by=[exog, 'index'], ascending=[False, True]);
                NCat = self.mOptions.mMaxExogenousCategories;
                lVC = lVC.head(NCat)
                # print("EXOGENOUS_DATA", exog, lVC.columns, lVC.head(NCat));
                lList = lVC['index'].values
                lList = sorted(lList.tolist());
                # print("most_frequent_categories_for" , exog, lList);
                if(len(lList) <= 1):
                    self.mExcluded.append(exog);
                else:
                    self.mExogenousVariableCategories[exog] = lList;
            else:
                self.mExogenousVariableCategories[exog] = None;
                stat_exog = (lEstimFrame[exog].mean(), lEstimFrame[exog].std());
                if(stat_exog[1] > 1e-5):
                    self.mContExogenousStats[exog] = stat_exog;
                else:
                    self.mExcluded.append(exog);
