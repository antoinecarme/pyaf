# Copyright (C) 2023 Antoine Carme <Antoine.Carme@outlook.com>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np

from . import Perf as tsperf
from . import SignalDecomposition_Trend as tstr
from . import SignalDecomposition_Cycle as tscy
from . import SignalDecomposition_AR as tsar
from . import Options as tsopts
from . import TimeSeriesModel as tsmodel
from . import TimeSeries_Cutting as tscut
from . import Utils as tsutil

import copy

class cModelSelector_Voting:
    def __init__(self):
        self.mOptions = None
        pass

    def dump_all_model_perfs_as_json(self):
        logger = tsutil.get_pyaf_logger();
        lColumns = ['Model', 'DetailedFormula' , 'Category', 'Complexity', 'Voting'] + [x for x in self.mTrPerfDetails.columns if x.startswith('Forecast_')]
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
        lVotingScores = self.compute_voting_scores(iSigDecs)
        lCriterion = self.mOptions.mModelSelection_Criterion
        rows_list = []
        lPerfsByModel = {}
        for (lName, sigdec) in iSigDecs.items():
            for (model , value) in sorted(sigdec.mPerfsByModel.items()):
                lPerfsByModel[model] = value
                lTranformName = sigdec.mSignal;
                lDecompType = model[1];
                lModelFormula = model
                tsmodel = value[0][2]
                lModelCategory = tsmodel.get_model_category()
                lSplit = value[0][2].mTimeInfo.mOptions.mCustomSplit
                #  value format : self.mPerfsByModel[lModel.mOutName] = [lModel, lComplexity, lFitPerf , lForecastPerf, lTestPerf];
                lComplexity = value[1];
                model_perfs = tsmodel.get_perfs_summary()
                lFitPerf = model_perfs["Fit"];
                lForecastPerf = model_perfs["Forecast"]
                lTestPerf = model_perfs["Test"]
                H = tsmodel.mTimeInfo.mHorizon

                lVoting = lVotingScores[ lModelFormula[3] ]
                row = [lSplit, lTranformName, lDecompType, lModelFormula[3], lModelFormula, lModelCategory, lComplexity,
                       lFitPerf[1].get(lCriterion),
                       lFitPerf[H].get(lCriterion), 
                       lForecastPerf[1].get(lCriterion),
                       lForecastPerf[H].get(lCriterion),
                       lTestPerf[1].get(lCriterion),
                       lTestPerf[H].get(lCriterion),
                       lVoting]
                rows_list.append(row);

        self.mTrPerfDetails =  pd.DataFrame(rows_list, columns=
                                            ('Split', 'Transformation', 'DecompositionType',
                                             'Model', 'DetailedFormula', 'Category', 'Complexity',
                                             'Fit_' + self.mOptions.mModelSelection_Criterion + "_1",
                                             'Fit_' + self.mOptions.mModelSelection_Criterion + "_H",
                                             'Forecast_' + self.mOptions.mModelSelection_Criterion + "_1",
                                             'Forecast_' + self.mOptions.mModelSelection_Criterion + "_H",
                                             'Test_' + self.mOptions.mModelSelection_Criterion + "_1",
                                             'Test_' + self.mOptions.mModelSelection_Criterion + "_H",
                                             "Voting")) 
        # print(self.mTrPerfDetails.head(self.mTrPerfDetails.shape[0]));
        lIndicator = 'Voting';
        lBestPerf = self.mTrPerfDetails[ lIndicator ].max();
        # allow a loss of one point (0.01 of MAPE) if complexity is reduced.
        assert(not np.isnan(lBestPerf))
        self.mTrPerfDetails.sort_values(by=[lIndicator, 'Complexity', 'Model'] ,
                                        ascending=[False, False, True],
                                        inplace=True);
        self.mTrPerfDetails = self.mTrPerfDetails.reset_index(drop=True);
        if(self.mOptions.mDebugPerformance):
            self.dump_all_model_perfs_as_json()
        lTol = 0.99
        lInterestingModels = self.mTrPerfDetails[self.mTrPerfDetails[lIndicator] >= (lBestPerf * lTol)].reset_index(drop=True);
        lInterestingModels.sort_values(by=['Complexity'] , ascending=[False], inplace=True)
        # print(self.mTransformList);
        # print(self.mPerfsByModel);
        lBestName = lInterestingModels['DetailedFormula'].iloc[0]
        lBestModel = lPerfsByModel[lBestName][0][2];
        # print("BEST_MODEL", lBestName, lBestModel)
        self.mBestModel = lBestModel
        self.mPerfsByModel = lPerfsByModel
        self.mModelShortList = lInterestingModels[['Transformation', 'DecompositionType', 'Model', lIndicator, 'Complexity', 'Forecast_' + self.mOptions.mModelSelection_Criterion + "_1",  'Forecast_' + self.mOptions.mModelSelection_Criterion + "_H"]] 
        # print(self.mModelShortList.head());
        return (iSignal, lPerfsByModel, lBestModel, self.mModelShortList)


    def perform_model_selection_cross_validation(self):
        lTimer = None
        if(self.mOptions.mDebugProfile):
            lTimer = tsutil.cTimer(("MODEL_SELECTION_FOR_CROSS_VALIDATION"))
        # self.mTrPerfDetails.to_csv("perf_time_series_cross_val.csv")
        lIndicator = [x for x in self.mTrPerfDetails.columns if x.startswith('Forecast_')][-1]
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
        lInterestingModels.sort_values(by=['Complexity'] , ascending=True, inplace=True)
        # print(self.mTransformList);
        # print(lInterestingModels.head());
        lBestName = lInterestingModels['DetailedFormula'].iloc[0];
        lBestSplit = lInterestingModels['Split'].iloc[0];
        self.mBestModel = self.mPerfsByModel[lBestName][0][2];
        self.mModelShortList = lInterestingModels[['Model', 'Category', 'Split', lIndicator, 'IC']]
        # print("BEST_MODEL", lBestName, self.mBestModel)


class cModelSelector_Condorcet(cModelSelector_Voting):
    
    def __init__(self):
        cModelSelector_Voting.__init__(self);
        self.mCondorcetScores = None
        self.mCriteriaValues = None

    def get_criterion_values(self, iModel):
        lCriterion = self.mOptions.mModelSelection_Criterion
        lPerfs1 = iModel.mForecastPerfs
        self.mHorizon = iModel.mTimeInfo.mHorizon
        lForecastColumn = str(iModel.mOriginalSignal) + "_Forecast";
        lCriterionValues = {}
        for h in range(self.mHorizon):
            lHorizonName = lForecastColumn + "_" + str(h + 1);
            lCriterionValues[h + 1] = lPerfs1[lHorizonName].get(lCriterion)
        return lCriterionValues
        
        
    def isBetter(self, iCrit1, iCrit2):
        # Adapt the condocrcet score for R2 (higher values are better).
        lHigherIsBetter = tsperf.cPerf.higher_values_are_better(self.mOptions.mModelSelection_Criterion)
        if(lHigherIsBetter):
            if(iCrit2 < (iCrit1 + 0.01)):
                return 1
        else:
            if(iCrit1 < (iCrit2 + 0.01)):
                return 1
        return 0

    def filter_worst_criteria_values(self):
        # Condorecet method will give the same result and run faster if we remove the worst models.
        lBestModels = []
        for h in range(self.mHorizon):
            lValues = [(k, v[h + 1]) for (k,v) in self.mCriteriaValues.items()]
            lValues = sorted(lValues, key = lambda x : x[1])
            lBestModels = lBestModels + [k[0] for k in lValues[:32]] # 32 best for each horizon
        lBestModels = set(lBestModels)
        lDiscrededModels = [x for x in self.mCriteriaValues.keys() if x not in lBestModels]
        print("KEPT_DISCARDED_MODELS", len(self.mCriteriaValues.keys()) , len(lBestModels), len(lDiscrededModels))
        for model_name in lDiscrededModels:
            self.mCriteriaValues.pop(model_name)
        
        
    def compute_condorcet_score(self, model_name):
        lCrit1 = self.mCriteriaValues.get(model_name)            
        if(lCrit1 is None):
            return 0
        lScore = 0
        for h in range(len(lCrit1)):
            lCoeff = (h + 1) / self.mHorizon
            for (model_name_2, lCrit2) in self.mCriteriaValues.items():
                if(model_name_2 != model_name):
                    lScore = lScore + self.isBetter(lCrit1[h + 1], lCrit2[h + 1]) * lCoeff
        lScore = round(lScore , 4)
        return lScore
        
    def compute_voting_scores(self, iSigDecs):
        self.mCondorcetScores = {}
        self.mCriteriaValues = {}
        lModels = {}
        for (lName, sigdec) in iSigDecs.items():
            for (model , value) in sorted(sigdec.mPerfsByModel.items()):
                ts_model = value[0][2]
                model_name = ts_model.mOutName
                lModels[model_name] = ts_model
                self.mCriteriaValues[model_name] = self.get_criterion_values(ts_model)
        # self.filter_worst_criteria_values()
        for (model_name, ts_model) in lModels.items():
            self.mCondorcetScores[model_name] = self.compute_condorcet_score(model_name)
        return self.mCondorcetScores


def create_model_selector(iMethod):
    return cModelSelector_Condorcet()
