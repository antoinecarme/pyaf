# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
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


    def to_json(self):
        dict1 = {};
        return dict1;

    def fit(self):
        self.mExogenousDataFrame = self.mExogenousData[0];
        self.mExogenousVariables = self.mExogenousData[1];
        self.mDateVariable = self.mTimeInfo.mTime;
        # print("preProcessExogenousVariables , columns", self.mExogenousVariables);
        self.updateExogenousVariableInfo();
        self.createEncodedExogenous();
        # print("preProcessExogenousVariables , dummy columns", self.mEncodedExogenous);

    def addVars(self, df):
        lExogDate = self.mDateVariable;
        # print(np.dtype(df[self.mDateVariable]) ,
        #      np.dtype(self.mEncodedExogenousDataFrame[self.mDateVariable]));
        # print(df.info());
        # print(self.mEncodedExogenousDataFrame.info());
        # print(df.head());
        # print(self.mEncodedExogenousDataFrame.head());

        df1 = df[[self.mDateVariable]];
        lCompleteEncoded = df1.merge(self.mEncodedExogenousDataFrame,
                                     how='left',
                                     left_on=self.mDateVariable,
                                     right_on=lExogDate);
        lCompleteEncoded.fillna(0.0, inplace=True);
        
        # print("EXOG_COLUMNS", self.mEncodedExogenousDataFrame.columns);
        df2 = df.merge(lCompleteEncoded,
                       how='left',
                       left_on=self.mDateVariable,
                       right_on=lExogDate);
        # df1 = df1.drop([lExogDate] , axis = 1);
        # print(df.head());
        return df2;

    def transformDataset(self, df):
        # print("BEFORE_EXOG_TRANSFORM_DATASET" , df.shape, df.columns);
        df1 = self.addVars(df);
        # print("AFTER_EXOG_TRANSFORM_DATASET" , df.shape, df.columns);
        return df1;
        
    def createEncodedExogenous(self):
        self.mExogDummiesDataFrame = pd.DataFrame();
        self.mEncodedExogenous = [];
        self.mEncodedExogenousDataFrame = pd.DataFrame();
        self.mEncodedExogenousDataFrame[self.mDateVariable] = self.mExogenousDataFrame[self.mDateVariable];
        for exog in self.mExogenousVariables:
            if(exog not in self.mExcluded):
                lList = self.mExogenousVariableCategories[exog];
                if(lList is not None):
                    for lCat in lList:
                        lDummyName = exog + "=" + str(lCat);
                        lVec = np.where(self.mExogenousDataFrame[exog] == lCat , 1, 0);
                        self.mEncodedExogenousDataFrame[lDummyName] = lVec;
                        self.mEncodedExogenous = self.mEncodedExogenous + [lDummyName];
                else:
                    lExogStats = self.mContExogenousStats[exog];
                    self.mEncodedExogenousDataFrame[exog] = (self.mExogenousDataFrame[exog] - lExogStats[0])/ lExogStats[1];
                    self.mEncodedExogenousDataFrame[exog].fillna(0.0, inplace=True);
                    self.mEncodedExogenous = self.mEncodedExogenous + [exog];
            else:
                print("EXCLUDED" , exog);


    def updateExogenousVariableInfo(self):
        self.mExogenousVariableCategories = {};
        self.mContExogenousStats = {};
        # Compute these stats only on the estimation part.
        lEstimFrame = self.mExogenousDataFrame[self.mExogenousDataFrame[self.mDateVariable] >= self.mTimeInfo.mTimeMin]
        lEstimFrame = lEstimFrame[lEstimFrame[self.mDateVariable] <= self.mTimeInfo.mTimeMax]
        for exog in self.mExogenousVariables:
            if(self.mExogenousDataFrame[exog].dtype == np.object):
                # use nan as a category
                lVC = lEstimFrame[exog].value_counts(dropna = False);
                NCat = self.mOptions.mMaxExogenousCategories;
                NCat = min(NCat , lVC.shape[0]);
                # print("EXOGENOUS_DATA", lVC.head(NCat));
                lList = [];
                for i in range(NCat):
                    if(lVC[i] > 5):
                        lList.append(lVC.index[i]);
                # lListlVC.index[0:NCat].tolist();
                # print("most_frequent_categories_for" , exog, lList);
                if(len(lList) == 0):
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
