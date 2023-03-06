# Copyright (C) 2023 Antoine Carme <Antoine.Carme@outlook.com>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np

from . import Perf as tsperf
from . import Utils as tsutil

import copy

class cModelSelector_OneSignal:
    def __init__(self):
        self.mOptions = None
        pass

    def dump_all_model_perfs_as_json(self):
        logger = tsutil.get_pyaf_logger();
        lColumns = ['Model', 'DetailedFormula' , 'Category', 'Complexity', 'Forecast' + self.mOptions.mModelSelection_Criterion]
        lPerf_df = self.mTrPerfDetails[lColumns].head(10)
        lDict = lPerf_df.to_dict('records')
        import json
        lPerfDump = json.dumps(lDict, default = lambda o: o.__dict__, indent=4, sort_keys=True);
        logger.info("PERF_DUMP_START")
        logger.info(lPerfDump)
        logger.info("PERF_DUMP_END")
        

    def collectPerformanceIndices_ModelSelection(self, iSignal, iSigDecs) :
        logger = tsutil.get_pyaf_logger();
        lTimer = tsutil.cTimer(("MODEL_SELECTION", {"Signal" : iSignal, "Transformations" : sorted(list(iSigDecs.keys()))}))
        rows_list = []
        lPerfsByModel = {}
        for (lName, sigdec) in iSigDecs.items():
            for (model , value) in sorted(sigdec.mPerfsByModel.items()):
                lPerfsByModel[model] = value
                lTranformName = sigdec.mSignal;
                lDecompType = model[1];
                lModelFormula = model
                lModelCategory = value[0][2].get_model_category()
                lSplit = value[0][2].mTimeInfo.mOptions.mCustomSplit
                #  value format : self.mPerfsByModel[lModel.mOutName] = [lModel, lComplexity, lFitPerf , lForecastPerf, lTestPerf];
                lComplexity = value[1];
                lFitPerf = value[2];
                lForecastPerf = value[3];
                lTestPerf = value[4];
                row = [lSplit, lTranformName, lDecompType, lModelFormula[3], lModelFormula, lModelCategory, lComplexity,
                       lFitPerf,
                       lForecastPerf,
                       lTestPerf]
                rows_list.append(row);

        self.mTrPerfDetails =  pd.DataFrame(rows_list, columns=
                                            ('Split', 'Transformation', 'DecompositionType',
                                             'Model', 'DetailedFormula', 'Category', 'Complexity',
                                             'Fit' + self.mOptions.mModelSelection_Criterion,
                                             'Forecast' + self.mOptions.mModelSelection_Criterion,
                                             'Test' + self.mOptions.mModelSelection_Criterion)) 
        # print(self.mTrPerfDetails.head(self.mTrPerfDetails.shape[0]));
        lIndicator = 'Forecast' + self.mOptions.mModelSelection_Criterion;
        lBestPerf = self.mTrPerfDetails[ lIndicator ].min();
        lHigherIsBetter = tsperf.cPerf.higher_values_are_better(self.mOptions.mModelSelection_Criterion)
        if(lHigherIsBetter):
            lBestPerf = self.mTrPerfDetails[ lIndicator ].max();
        # allow a loss of one point (0.01 of MAPE) if complexity is reduced.
        assert(not np.isnan(lBestPerf))
        self.mTrPerfDetails.sort_values(by=[lIndicator, 'Complexity', 'Model'] ,
                                        ascending=[not lHigherIsBetter, False, True],
                                        inplace=True);
        self.mTrPerfDetails = self.mTrPerfDetails.reset_index(drop=True);
        if(self.mOptions.mDebugPerformance):
            self.dump_all_model_perfs_as_json()
        if(lHigherIsBetter):
            lInterestingModels = self.mTrPerfDetails[self.mTrPerfDetails[lIndicator] >= (lBestPerf - 0.01)].reset_index(drop=True);
        else:
            lInterestingModels = self.mTrPerfDetails[self.mTrPerfDetails[lIndicator] <= (lBestPerf + 0.01)].reset_index(drop=True);
            
        lInterestingModels.sort_values(by=['Complexity'] , ascending=False, inplace=True)
        # print(self.mTransformList);
        # print(lInterestingModels.head());
        # print(self.mPerfsByModel);
        lBestName = lInterestingModels['DetailedFormula'].iloc[0]
        lBestModel = lPerfsByModel[lBestName][0][2];
        # print("BEST_MODEL", lBestName, lBestModel)
        self.mBestModel = lBestModel
        self.mPerfsByModel = lPerfsByModel
        self.mModelShortList = lInterestingModels[['Transformation', 'DecompositionType', 'Model', lIndicator, 'Complexity']] 
        return (iSignal, lPerfsByModel, lBestModel, self.mModelShortList)


    def perform_model_selection_cross_validation(self):
        lTimer = None
        if(self.mOptions.mDebugProfile):
            lTimer = tsutil.cTimer(("MODEL_SELECTION_FOR_CROSS_VALIDATION"))
        # self.mTrPerfDetails.to_csv("perf_time_series_cross_val.csv")
        lIndicator = 'Forecast' + self.mOptions.mModelSelection_Criterion;
        lColumns = ['Category', 'Complexity', lIndicator]
        lPerfByCategory = self.mTrPerfDetails[lColumns].groupby(by=['Category'] , sort=False)[lIndicator].mean()
        lPerfByCategory_df = pd.DataFrame(lPerfByCategory).reset_index()
        lPerfByCategory_df.columns = ['Category' , lIndicator]
        print("CROSS_VAL_PERF", lPerfByCategory_df)
        # lPerfByCategory_df.to_csv("perf_time_series_cross_val_by_category.csv")
        lBestPerf = lPerfByCategory_df[ lIndicator ].min();
        lHigherIsBetter = tsperf.cPerf.higher_values_are_better(self.mOptions.mModelSelection_Criterion)
        if(lHigherIsBetter):
            lBestPerf = lPerfByCategory_df[ lIndicator ].max();
        lPerfByCategory_df.sort_values(by=[lIndicator, 'Category'] ,
                                ascending=[not lHigherIsBetter, True],
                                inplace=True);
        lPerfByCategory_df = lPerfByCategory_df.reset_index(drop=True);
                
        if(lHigherIsBetter):
            lInterestingCategories_df = lPerfByCategory_df[lPerfByCategory_df[lIndicator] >= (lBestPerf - 0.01)].reset_index(drop=True);
        else:
            lInterestingCategories_df = lPerfByCategory_df[lPerfByCategory_df[lIndicator] <= (lBestPerf + 0.01)].reset_index(drop=True);
            
        # print(lPerfByCategory_df.head());
        # print(lInterestingCategories_df.head());
        # print(self.mPerfsByModel);
        lInterestingCategories = list(lInterestingCategories_df['Category'].unique())
        self.mTrPerfDetails['IC'] = self.mTrPerfDetails['Category'].apply(lambda x :1 if x in lInterestingCategories else 0) 
        lInterestingModels = self.mTrPerfDetails[self.mTrPerfDetails['IC'] == 1].copy()
        lInterestingModels.sort_values(by=['Complexity'] , ascending=False, inplace=True)
        # print(self.mTransformList);
        # print(lInterestingModels.head());
        lBestName = lInterestingModels['DetailedFormula'].iloc[0];
        lBestSplit = lInterestingModels['Split'].iloc[0];
        self.mBestModel = self.mPerfsByModel[lBestName][0][2];
        self.mModelShortList = lInterestingModels[['Model', 'Category', 'Split', lIndicator, 'IC']]
        # print("BEST_MODEL", lBestName, self.mBestModel)


def create_model_selector():
    return cModelSelector_OneSignal()
