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
        lExogDate = "exog_date_" + self.mDateVariable;
        # print("EXOG_COLUMNS", self.mEncodedExogenousDataFrame.columns);
        df = df.merge(self.mEncodedExogenousDataFrame,
                      how='left',
                      left_on=self.mDateVariable,
                      right_on=lExogDate);
        df = df.drop([lExogDate] , axis = 1);
        return df;

    def transformDataset(self, df):
        # print("BEFORE_EXOG_TRANSFORM_DATASET" , df.shape, df.columns);
        df = self.addVars(df);
        # print("AFTER_EXOG_TRANSFORM_DATASET" , df.shape, df.columns);
        return df;
        
    def createEncodedExogenous(self):
        self.mExogDummiesDataFrame = pd.DataFrame();
        self.mEncodedExogenous = [];
        self.mEncodedExogenousDataFrame = pd.DataFrame();
        self.mEncodedExogenousDataFrame["exog_date_" + self.mDateVariable] = self.mExogenousDataFrame[self.mDateVariable];
        for exog in self.mExogenousVariables:
            lList = self.mExogenousVariableCategories[exog];
            if(lList is not None):
                for lCat in lList:
                    lDummyName = "exog_dummy_" + exog + "=" + str(lCat);
                    self.mEncodedExogenousDataFrame[lDummyName] = self.mExogenousDataFrame[exog].apply(lambda x : 1 if (x == lCat) else 0);
                    self.mEncodedExogenous = self.mEncodedExogenous + [lDummyName];
            else:
                lExogStats = self.mContExogenousStats[exog];
                self.mEncodedExogenousDataFrame[exog] = (self.mExogenousDataFrame[exog] - lExogStats[0])/ lExogStats[1];
                self.mEncodedExogenousDataFrame[exog].fillna(0.0, inplace=True);
                self.mEncodedExogenous = self.mEncodedExogenous + [exog];


    def updateExogenousVariableInfo(self):
        self.mExogenousVariableCategories = {};
        self.mContExogenousStats = {};
        lEstimFrame = self.mTimeInfo.getEstimPart(self.mExogenousDataFrame)
        for exog in self.mExogenousVariables:
            if(self.mExogenousDataFrame[exog].dtype == np.object):
                # use nan as a category
                lVC = lEstimFrame[exog].value_counts(dropna = False);
                NCat = self.mOptions.mMaxExogenousCategories;
                lList = lVC.index[0:NCat].tolist();
                print("most_frequent_categories_for" , exog, lList);
                self.mExogenousVariableCategories[exog] = lList;
            else:
                self.mExogenousVariableCategories[exog] = None;
                self.mContExogenousStats[exog] = (lEstimFrame[exog].mean(), lEstimFrame[exog].std());
