import pandas as pd
import numpy as np


from . import Utils as tsutil

class cExogenousInfo:

    def __init__(self):
        self.mExogenousVariables = None;
        self.mExogenousDummies = None;
        self.mExogenousVariableCategories = None;
        self.mExogenousDataFrame = None;
        self.mExogenousData = None;        
        self.mDateVariable = None;
        
    def info(self):
        lStr2 = "ExogenousVariables = '" + self.mExogD +"'";
        lStr2 += " TimeMin=" + str(self.mTimeMin) +"";
        return lStr2;


    def to_json(self):
        dict1 = {};
        return dict1;

    def fit(self):
        self.mExogenousDataFrame = self.mExogenousData[0];
        self.mExogenousVariables = self.mExogenousData[1];
        # print("preProcessExogenousVariables , columns", self.mExogenousVariables);
        self.updateExogenousVariableInfo();
        self.createExogenousDummies();
        # print("preProcessExogenousVariables , dummy columns", self.mExogenousDummies);

    def addVars(self, df):
        lExogDate = "exog_date_" + self.mDateVariable;
        # print("EXOG_COLUMNS", self.mExogenousDummiesDataFrame.columns);
        df = df.merge(self.mExogenousDummiesDataFrame,
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

    def updateFutureExogenousDummies(self, df):
        for exog in self.mExogenousVariables:
            lList = self.mExogenousVariableCategories[exog];
            for lCat in lList:
                lDummyName = "exog_dummy_" + exog + "=" + str(lCat);
                lInputDS[lDummyName] = iInputDS[exog].apply(lambda x : 1 if (x == lCat) else 0);
        return df;

        
    def createExogenousDummies(self):
        self.mExogDummiesDataFrame = pd.DataFrame();
        self.mExogenousDummies = [];
        self.mExogenousDummiesDataFrame = pd.DataFrame();
        self.mExogenousDummiesDataFrame["exog_date_" + self.mDateVariable] = self.mExogenousDataFrame[self.mDateVariable];
        for exog in self.mExogenousVariables:
            lList = self.mExogenousVariableCategories[exog];
            for lCat in lList:
                lDummyName = "exog_dummy_" + exog + "=" + str(lCat);
                self.mExogenousDummiesDataFrame[lDummyName] = self.mExogenousDataFrame[exog].apply(lambda x : 1 if (x == lCat) else 0);
                self.mExogenousDummies = self.mExogenousDummies + [lDummyName];


    def updateExogenousVariableInfo(self):
        self.mExogenousVariableCategories = {};
        for exog in self.mExogenousVariables:
            # use nan as a category
            lVC = self.mExogenousDataFrame[exog].value_counts(dropna = False);
            # TODO : 5 categories. to be added in options.
            NCat = 5;
            lList = lVC.index[0:NCat].tolist();
            print("most_frequent_categories_for" , exog, lList);
            self.mExogenousVariableCategories[exog] = lList;

