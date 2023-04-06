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


    def collectPerformanceIndices_ModelSelection(self, iSignal, iPerfsByModel) :
        logger = tsutil.get_pyaf_logger();
        lTimer = tsutil.cTimer(("VOTING_MODEL_SELECTION", {"Signal" : iSignal}))
        lVotingScores = self.compute_voting_scores(iPerfsByModel)
        lCriterion = self.mOptions.mModelSelection_Criterion
        rows_list = []
        for (model , value) in sorted(iPerfsByModel.items()):
            lTranformName = value["Signal"]
            lDecompType = value["DecompositionType"];
            lModelFormula = model
            modelname = value["ModelName"]
            lModelCategory = value["ModelCategory"]
            lSplit = value["Split"]
            #  value format : self.mPerfsByModel[lModel.mOutName] = [lModel, lComplexity, lFitPerf , lForecastPerf, lTestPerf];
            lComplexity = value["Complexity"];
            lOriginalSignal = value["OriginalSignal"]
            lForecastColumn = str(lOriginalSignal) + "_Forecast";
            lFitPerf = value["FitPerf"];
            lForecastPerf = value["ForecastPerf"]
            lTestPerf = value["TestPerf"]
            H = value["Horizon"]

            lVoting = lVotingScores[ modelname ]
            row = [lSplit, lTranformName, lDecompType, lModelFormula[3], lModelFormula, lModelCategory, lComplexity]
            for h in range(1, H+1):
                row = row + [lFitPerf[lForecastColumn + "_" + str(h)].get(lCriterion),
                             lForecastPerf[lForecastColumn + "_" + str(h)].get(lCriterion),
                             lTestPerf[lForecastColumn + "_" + str(h)].get(lCriterion)]
            row = row + [ lVoting ]
            rows_list.append(row);

        Cols = ['Split', 'Transformation', 'DecompositionType',
                'Model', 'DetailedFormula', 'Category', 'Complexity']
        for h in range(1, H+1):
            Cols = Cols + ['Fit_' + self.mOptions.mModelSelection_Criterion + "_" + str(h),
                           'Forecast_' + self.mOptions.mModelSelection_Criterion + "_" + str(h),
                           'Test_' + self.mOptions.mModelSelection_Criterion + "_" + str(h)]
            
        Cols = Cols + ["Voting"]
        self.mTrPerfDetails =  pd.DataFrame(rows_list, columns= Cols)
        # tsutil.print_pyaf_detailed_info(self.mTrPerfDetails.head(self.mTrPerfDetails.shape[0]));
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
        # tsutil.print_pyaf_detailed_info(self.mTransformList);
        # tsutil.print_pyaf_detailed_info(self.mPerfsByModel);
        self.mBestModelName = lInterestingModels['DetailedFormula'].iloc[0]
        # tsutil.print_pyaf_detailed_info("BEST_MODEL", lBestName, lBestModel)
        self.mModelShortList = lInterestingModels[['Transformation', 'DecompositionType', 'Model', lIndicator, 'Complexity'] + ['Forecast_' + self.mOptions.mModelSelection_Criterion + "_" + str(h) for h in range(1, H+1)]] 
        # tsutil.print_pyaf_detailed_info(self.mModelShortList.head());
        return (iSignal, self.mBestModelName, self.mModelShortList)


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
        # lPerfByCategory_df.to_csv("perf_time_series_cross_val_by_category.csv")
        lBestPerf = lPerfByCategory_df[ lIndicator ].min();
        lHigherIsBetter = tsperf.cPerf.higher_values_are_better(self.mOptions.mModelSelection_Criterion)
        if(lHigherIsBetter):
            lBestPerf = lPerfByCategory_df[ lIndicator ].max();
        lPerfByCategory_df.sort_values(by=[lIndicator, 'Category'] ,
                                ascending=[not lHigherIsBetter, True],
                                inplace=True);
        lPerfByCategory_df = lPerfByCategory_df.reset_index(drop=True);
        tsutil.print_pyaf_detailed_info("CROSS_VAL_PERF", [row for row in lPerfByCategory_df.itertuples(name='CV')][:5])
                
        if(lHigherIsBetter):
            lInterestingCategories_df = lPerfByCategory_df[lPerfByCategory_df[lIndicator] >= (lBestPerf - 0.01)].reset_index(drop=True);
        else:
            lInterestingCategories_df = lPerfByCategory_df[lPerfByCategory_df[lIndicator] <= (lBestPerf + 0.01)].reset_index(drop=True);
            
        # tsutil.print_pyaf_detailed_info(lPerfByCategory_df.head());
        # tsutil.print_pyaf_detailed_info(lInterestingCategories_df.head());
        # tsutil.print_pyaf_detailed_info(self.mPerfsByModel);
        lInterestingCategories = list(lInterestingCategories_df['Category'].unique())
        self.mTrPerfDetails['IC'] = self.mTrPerfDetails['Category'].apply(lambda x :1 if x in lInterestingCategories else 0) 
        lInterestingModels = self.mTrPerfDetails[self.mTrPerfDetails['IC'] == 1].copy()
        lInterestingModels.sort_values(by=['Complexity'] , ascending=True, inplace=True)
        # tsutil.print_pyaf_detailed_info(self.mTransformList);
        # tsutil.print_pyaf_detailed_info(lInterestingModels.head());
        lBestName = lInterestingModels['DetailedFormula'].iloc[0];
        lBestSplit = lInterestingModels['Split'].iloc[0];
        self.mBestModelName = lBestName
        self.mModelShortList = lInterestingModels[['Model', 'Category', 'Split', lIndicator, 'IC']]
        # tsutil.print_pyaf_detailed_info("BEST_MODEL", lBestName, self.mBestModel)


class cModelSelector_Condorcet(cModelSelector_Voting):
    
    def __init__(self):
        cModelSelector_Voting.__init__(self);
        self.mCondorcetScores = None
        self.mCriteriaValues = None
        self.mMaxCandidates = 128

    def get_criterion_values(self, value):
        lCriterion = self.mOptions.mModelSelection_Criterion
        lPerfs1 = value["ForecastPerf"]
        self.mHorizon = value["Horizon"]
        lForecastColumn = value["OriginalSignal"] + "_Forecast";
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
            lBestModels = lBestModels + [k[0] for k in lValues[:self.mMaxCandidates]] # self.mMaxCandidates best for each horizon
        lBestModels = set(lBestModels)
        lDiscrededModels = [x for x in self.mCriteriaValues.keys() if x not in lBestModels]
        tsutil.print_pyaf_detailed_info("KEPT_DISCARDED_MODELS", len(self.mCriteriaValues.keys()) , len(lBestModels), len(lDiscrededModels))
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
        
    def compute_voting_scores(self, iPerfsByModel):
        lTimer = tsutil.cTimer(("MODEL_SELECTION_COMPUTE_VOTING_SCORES", len(iPerfsByModel)))
        self.mCondorcetScores = {}
        self.mCriteriaValues = {}
        lModels = {}
        for (model , value) in sorted(iPerfsByModel.items()):
            model_name = value["ModelName"]
            lModels[model_name] = value
            self.mCriteriaValues[model_name] = self.get_criterion_values(value)
        if(len(self.mCriteriaValues) > self.mMaxCandidates):
            self.filter_worst_criteria_values()
        for (model_name, ts_model) in lModels.items():
            self.mCondorcetScores[model_name] = self.compute_condorcet_score(model_name)
        return self.mCondorcetScores


def create_model_selector(iMethod):
    return cModelSelector_Condorcet()
