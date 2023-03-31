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

    def long_list_to_str(self, iList):
        lList_str = ""
        if(len(iList) > 10):
            lList_str = str(iList[:5]) + " ... " + str(iList[-5:])
        else:
            lList_str = str(iList)
        return lList_str

    def decode_used_vars(self, used):
        used_vars = used
        if(used is not None):
            used_vars = [x.split("=")[0] for x in used]
            used_vars = [x for x in used_vars if x in self.mExogenousVariables]
            used_vars = list(set(used_vars))
        return used_vars


    def info(self, used = None):
        logger = tsutil.get_pyaf_logger();
        lDict = self.to_dict(used)
        logger.info("EXOGENOUS_VARIABLE_DETAIL_START")
        for exog in lDict["Categorical_Variables"].keys():
            lList = lDict["Categorical_Variables"][exog]
            lUsed_cat = lDict["Categorical_Variables_Usage"][exog]
            logger.info("EXOGENOUS_VARIABLE_DETAIL_CATEGORICAL_FREQUENCIES '" + exog + "' " + str(lList))
            logger.info("EXOGENOUS_VARIABLE_DETAIL_CATEGORICAL_USED '" + exog + "' " + str(lUsed_cat))
        lExcluded_cat = lDict["Categorical_Variables_Excluded"]
        logger.info("EXOGENOUS_VARIABLE_DETAIL_CATEGORICAL_EXCLUDED " + str(len(lExcluded_cat)) + " " + self.long_list_to_str(lExcluded_cat))
        for (exog, lStats) in lDict["Continuous_Variables"].items():
            logger.info("EXOGENOUS_VARIABLE_DETAIL_CONTINUOUS '" + exog + "' " + str(lStats))
        lExcluded_cont = lDict["Continuous_Variables_Excluded"]
        logger.info("EXOGENOUS_VARIABLE_DETAIL_CONTINUOUS_EXCLUDED " + str(len(lExcluded_cont)) + " " + self.long_list_to_str(lExcluded_cont))
        
        logger.info("EXOGENOUS_VARIABLE_DETAIL_END")


    def to_dict(self, used = None):
        used_vars = self.decode_used_vars(used)
        dict1 = {};
        lExcluded_cat = []
        dict1 ["Categorical_Variables"] = {}
        dict1 ["Categorical_Variables_Usage"] = {}
        for (exog, lList) in self.mExogenousVariableCategories.items():
            if(used_vars is not None and (exog in used_vars)):
                lUsed_cat = []
                for (cat, freq) in lList:
                    lUsed_cat_i = [x for x in used if x.startswith(exog + "=" + str(cat))]
                    if(len(lUsed_cat_i) > 0):
                        lUsed_cat = lUsed_cat + [exog + "=" + str(cat)]
                dict1 ["Categorical_Variables"][exog] = dict([(np.array([x[0]]).item(), x[1].item()) for x in lList])
                dict1 ["Categorical_Variables_Usage"][exog] = lUsed_cat
            else:
                lExcluded_cat = lExcluded_cat + [exog]
        dict1 ["Categorical_Variables_Excluded"] = lExcluded_cat
        dict1 ["Continuous_Variables"] = {}
        lExcluded_cont = []
        for exog in self.mContExogenousStats.keys():
            if(used_vars is not None and (exog in used_vars)):
                lExogStats = self.mContExogenousStats[exog];
                dict1 ["Continuous_Variables"][exog] = {"Mean" : lExogStats[0], "StdDev" : lExogStats[1]}
            else:
                lExcluded_cont = lExcluded_cont + [exog]
        dict1 ["Continuous_Variables_Excluded"] = lExcluded_cont
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
        # tsutil.print_pyaf_detailed_info("preProcessExogenousVariables , columns", self.mExogenousVariables);
        self.updateExogenousVariableInfo();
        self.createEncodedExogenous();
        # tsutil.print_pyaf_detailed_info("preProcessExogenousVariables , dummy columns", self.mEncodedExogenous);

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

    def encode_exogenous_categorical(self, exog):
        lSeries = self.mExogenousDataFrame[exog]
        lList = self.mExogenousVariableCategories[exog];
        lEncodedVars = {}
        assert(lList is not None)
        for lCat in lList:
            lDummyName = exog + "=" + str(lCat[0]);
            lEncodedVars[lDummyName] = np.where(lSeries == lCat[0] , np.int8(1), np.int8(0));
        return lEncodedVars
        
    def encode_exogenous_continuous(self, exog):
        lSeries = self.mExogenousDataFrame[exog]
        lEncodedVars = {}
        lExogStats = self.mContExogenousStats[exog];
        # single precision here ...
        lStandardized = (lSeries - lExogStats[0])/ lExogStats[1];
        lStandardized = lStandardized.astype(np.float32);
        lStandardized.fillna(np.float32(0.0), inplace=True);
        lEncodedVars[exog] = lStandardized
        return lEncodedVars

    
    def createEncodedExogenous(self):
        self.mEncodedExogenous = [];
        lEncodedVars = {self.mDateVariable : self.mExogenousDataFrame[self.mDateVariable]}
        lEncodedVars_exog = {}
        for exog in self.mExogenousVariableCategories.keys():
            lEncodedVars_exog_cat = self.encode_exogenous_categorical(exog)
            lEncodedVars_exog.update(lEncodedVars_exog_cat)
        for exog in self.mContExogenousStats.keys():
            lEncodedVars_exog_cont = self.encode_exogenous_continuous(exog)
            lEncodedVars_exog.update(lEncodedVars_exog_cont)
        for(lName, lEncodedSeries) in lEncodedVars_exog.items():
            lEncodedVars[lName] = lEncodedSeries
            self.mEncodedExogenous = self.mEncodedExogenous + [lName];
        self.mEncodedExogenousDataFrame = pd.DataFrame(lEncodedVars,
                                                       index = self.mExogenousDataFrame.index);
        # self.mEncodedExogenousDataFrame.info()

    def updateExogenousVariableInfo_Categorical(self, exog, series):
        NCat = self.mOptions.mMaxExogenousCategories;
        # use nan as a category
        lValues, lCounts = np.unique(series, return_counts=True)
        lValueCounts = zip([x for x in lValues], [x for x in lCounts])
        lValueCounts_Significant = [x for x in lValueCounts if(x[1] > 5)]
        lValueCounts_ordered = sorted(lValueCounts_Significant, key = lambda x : (-x[1], x[0]))
        lList = lValueCounts_ordered[:NCat]
        lList = [(x[0], x[1]) for x in lList]
        if(len(lList) <= 1):
            self.mExcluded.append(exog);
        else:
            self.mExogenousVariableCategories[exog] = lList;
        
    def updateExogenousVariableInfo_Continuous(self, exog, series):
        stat_exog = (series.mean(), series.std());
        if(stat_exog[1] > 1e-5):
            self.mContExogenousStats[exog] = stat_exog;
        else:
            self.mExcluded.append(exog);
        
        
    def updateExogenousVariableInfo(self):
        # self.mExogenousDataFrame.info()
        # tsutil.print_pyaf_detailed_info(self.mExogenousDataFrame.describe());
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
            # tsutil.print_pyaf_detailed_info("EXOG_DTYPE" , exog, lType)
            if((lType == object) or (lType.name == 'category')):
                self.updateExogenousVariableInfo_Categorical(exog, lEstimFrame[exog])
            else:
                self.updateExogenousVariableInfo_Continuous(exog, lEstimFrame[exog])
