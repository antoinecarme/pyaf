# Copyright (C) 2016 Antoine Carme <Antoine.Carme@outlook.com>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np


from . import Time as tsti
from . import Exogenous as tsexog
from . import MissingData as tsmiss
from . import Signal_Transformation as tstransf
from . import Perf as tsperf
from . import SignalDecomposition_Trend as tstr
from . import SignalDecomposition_Cycle as tscy
from . import SignalDecomposition_AR as tsar
from . import Options as tsopts
from . import TimeSeriesModel as tsmodel
from . import TimeSeries_Cutting as tscut
from . import Utils as tsutil

import copy

def sample_signal_if_needed(iInputDS, iOptions):
    logger = tsutil.get_pyaf_logger();
    lInputDS = iInputDS
    if(iOptions.mActivateSampling):
        if(iOptions.mDebugProfile):
            logger.info("PYAF_MODEL_SAMPLING_ACTIVATED " +
                        str((iOptions.mSamplingThreshold, iOptions.mSeed)));
        lInputDS = iInputDS.tail(iOptions.mSamplingThreshold);
    return lInputDS

class cSignalDecompositionOneTransform:
        
    def __init__(self):
        self.mSignalFrame = None
        self.mTime = None
        self.mSignal = None
        self.mTimeInfo = tsti.cTimeInfo();
        self.mTransformation = tstransf.cSignalTransform_None();

        
    def setParams(self , iInputDS, iTime, iSignal, iHorizon, iTransformation,
                  iDecomspositionType, iExogenousData = None):
        assert(iInputDS.shape[0] > 0)
        assert(iInputDS.shape[1] > 0)
        assert(iTime in iInputDS.columns)
        assert(iSignal in iInputDS.columns)

        # print("setParams , head", iInputDS.head());
        # print("setParams , tail", iInputDS.tail());
        # print("setParams , columns", iInputDS.columns);
        
        self.mTime = iTime
        self.mOriginalSignal = iSignal;

        self.mDecompositionType = iDecomspositionType
        
        self.mTransformation = iTransformation;
        self.mTransformation.mOriginalSignal = iSignal; 
        self.mTransformation.mOptions = self.mOptions;

        self.mSignal = iTransformation.get_name(iSignal)
        self.mHorizon = iHorizon;



        self.mSplit = tscut.cCuttingInfo()
        self.mSplit.mTime = self.mTime;
        self.mSplit.mSignal = self.mSignal;
        self.mSplit.mOriginalSignal = self.mOriginalSignal;
        self.mSplit.mHorizon = self.mHorizon;
        self.mSplit.mOptions = self.mOptions;
        
        
        self.mTimeInfo = tsti.cTimeInfo();
        self.mTimeInfo.mTime = self.mTime;
        self.mTimeInfo.mSignal = self.mSignal;
        self.mTimeInfo.mOriginalSignal = self.mOriginalSignal;
        self.mTimeInfo.mHorizon = self.mHorizon;
        self.mTimeInfo.mOptions = self.mOptions;
        self.mTimeInfo.mSplit = self.mSplit;

        self.mExogenousInfo = None;
        if(iExogenousData is not None):
            self.mExogenousInfo = tsexog.cExogenousInfo();
            self.mExogenousInfo.mExogenousData = iExogenousData;
            self.mExogenousInfo.mTimeInfo = self.mTimeInfo;
            self.mExogenousInfo.mOptions = self.mOptions;
        


    def updatePerfsForAllModels(self , iModels):
        self.mPerfsByModel = {}
        for model in iModels.keys():
            iModels[model].updatePerfs();
            
        for (name, model) in iModels.items():
            # print(name, model.__dict__);
            lComplexity = model.getComplexity();
            lFitPerf = model.mFitPerf;
            lForecastPerf = model.mForecastPerf;
            lTestPerf = model.mTestPerf;
            lSplit = model.mTimeInfo.mOptions.mCustomSplit
            self.mPerfsByModel[(self.mSignal, self.mDecompositionType, lSplit, model.mOutName)] = [(self.mSignal, self.mDecompositionType, model), lComplexity, lFitPerf , lForecastPerf, lTestPerf];
        return iModels;


    def train(self , iInputDS, iSplit, iTime, iSignal,
              iHorizon, iTransformation, iDecomspositionType):
        lTimer = tsutil.cTimer(("TRAINING", {"Signal" : iSignal, "Transformation" : iTransformation.get_name(iSignal)}))
        lInputDS = iInputDS[[iTime, iSignal]].copy()
        lInputDS = sample_signal_if_needed(lInputDS, self.mOptions)
        
        self.setParams(lInputDS, iTime, iSignal, iHorizon, iTransformation, iDecomspositionType, self.mExogenousData);

        lMissingImputer = tsmiss.cMissingDataImputer()
        lMissingImputer.mOptions = self.mOptions
        self.mSignalFrame = lMissingImputer.apply(lInputDS, iTime, iSignal).copy()
        assert(self.mSignalFrame.shape[0] > 0)
            
        # estimate time info
        # assert(self.mTimeInfo.mSignalFrame.shape[0] == iInputDS.shape[0])
        self.mSplit.mSignalFrame = self.mSignalFrame;
        self.mSplit.estimate();
        self.mTimeInfo.mSignalFrame = self.mSignalFrame;
        self.mTimeInfo.estimate();
        self.mSignalFrame['row_number'] = np.arange(0, self.mSignalFrame.shape[0]);

        lSignal = self.mSignalFrame[self.mOriginalSignal]
        self.mTransformation.fit(lSignal);

        self.mSignalFrame[self.mSignal] = self.mTransformation.apply(lSignal);
        # self.mSignalFrame[self.mSignal] = self.mSignalFrame[self.mSignal].astype(np.float32);
        
        if(self.mExogenousInfo is not None):
            lTimer2 = None
            if(self.mOptions.mDebugProfile):
                lTimer2 = tsutil.cTimer(("TRAINING_EXOGENOUS_DATA", {"Signal" : iSignal}))
            self.mExogenousInfo.fit();

        # estimate the trend

        lTrendEstimator = tstr.cTrendEstimator()
        lTrendEstimator.mSignalFrame = self.mSignalFrame
        lTrendEstimator.mTimeInfo = self.mTimeInfo
        lTrendEstimator.mSplit = self.mSplit
        lTrendEstimator.mOptions = self.mOptions;
        lTrendEstimator.mDecompositionType = iDecomspositionType
        
        lTrendEstimator.estimateTrend();
        #lTrendEstimator.plotTrend();


        # estimate cycles

        lCycleEstimator = tscy.cCycleEstimator();
        lCycleEstimator.mTrendFrame = lTrendEstimator.mTrendFrame;
        lCycleEstimator.mTrendList = lTrendEstimator.mTrendList;

        del lTrendEstimator;

        lCycleEstimator.mTimeInfo = self.mTimeInfo
        lCycleEstimator.mSplit = self.mSplit
        lCycleEstimator.mDecompositionType = iDecomspositionType
        lCycleEstimator.mOptions = self.mOptions;

        lCycleEstimator.estimateAllCycles();


        # autoregressive
        lAREstimator = tsar.cAutoRegressiveEstimator();
        lAREstimator.mCycleFrame = lCycleEstimator.mCycleFrame;
        lAREstimator.mTrendList = lCycleEstimator.mTrendList;
        lAREstimator.mCycleList = lCycleEstimator.mCycleList;

        del lCycleEstimator;

        lAREstimator.mTimeInfo = self.mTimeInfo
        lAREstimator.mSplit = self.mSplit
        lAREstimator.mDecompositionType = iDecomspositionType
        lAREstimator.mExogenousInfo = self.mExogenousInfo;
        lAREstimator.mOptions = self.mOptions;
        lAREstimator.estimate();

        # forecast perfs
        lModels = {};
        for trend in lAREstimator.mTrendList:
            for cycle in lAREstimator.mCycleList[trend]:
                cycle_residue = cycle.getCycleResidueName();
                for autoreg in lAREstimator.mARList[cycle_residue]:
                    lModel = tsmodel.cTimeSeriesModel(self.mTransformation, self.mDecompositionType,
                                                      trend, cycle, autoreg);
                    lModels[lModel.mOutName] = lModel;

        del lAREstimator;
        self.updatePerfsForAllModels(lModels);
        

class cTraining_Arg:
    def __init__(self , name):
        self.mName = name;
        self.mInputDS = None;
        self.mTime = None;
        self.mSignal = None;
        self.mHorizon = None;
        self.mTransformation = None;
        self.mSigDec = None;
        self.mSplit = None
        self.mResult = None;


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
                       lFitPerf.getCriterionValue(self.mOptions.mModelSelection_Criterion),
                       lForecastPerf.getCriterionValue(self.mOptions.mModelSelection_Criterion),
                       lTestPerf.getCriterionValue(self.mOptions.mModelSelection_Criterion)]
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
        # allow a loss of one point (0.01 of MAPE) if complexity is reduced.
        assert(not np.isnan(lBestPerf))
        self.mTrPerfDetails.sort_values(by=[lIndicator, 'Complexity', 'Model'] ,
                                        ascending=[True, True, True],
                                        inplace=True);
        self.mTrPerfDetails = self.mTrPerfDetails.reset_index(drop=True);
        if(self.mOptions.mDebugPerformance):
            self.dump_all_model_perfs_as_json()
                
        lInterestingModels = self.mTrPerfDetails[self.mTrPerfDetails[lIndicator] <= (lBestPerf + 0.01)].reset_index(drop=True);
        lInterestingModels.sort_values(by=['Complexity'] , ascending=True, inplace=True)
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

    def collectPerformanceIndices(self, iSignal, iSigDecs) :
        logger = tsutil.get_pyaf_logger();
        lTimer = None
        if(self.mOptions.mDebugProfile):
            lTimer = tsutil.cTimer(("MODEL_SELECTION", {"Signal" : iSignal}))

        rows_list = []
        lPerfsByModel = {}
        for (lName, sigdec) in iSigDecs.items():
            for (model , value) in sorted(sigdec.mPerfsByModel.items()):
                lPerfsByModel[model] = value;
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
                row = [lSplit, lTranformName, lDecompType, lModelFormula[3], lModelFormula , lModelCategory, lComplexity,
                       lFitPerf.mCount, lFitPerf.mL1, lFitPerf.mL2, lFitPerf.mMAPE,  lFitPerf.mMASE, lFitPerf.mCRPS, 
                       lForecastPerf.mCount, lForecastPerf.mL1, lForecastPerf.mL2, lForecastPerf.mMAPE, lForecastPerf.mMASE, lForecastPerf.mCRPS,
                       lTestPerf.mCount, lTestPerf.mL1, lTestPerf.mL2, lTestPerf.mMAPE, lTestPerf.mMASE, lTestPerf.mCRPS]
                rows_list.append(row);

        self.mTrPerfDetails =  pd.DataFrame(rows_list, columns=
                                            ('Split', 'Transformation', 'DecompositionType', 'Model',
                                             'DetailedFormula', 'Category', 'Complexity',
                                             'FitCount', 'FitL1', 'FitL2', 'FitMAPE', 'FitMASE', 'FitCRPS',
                                             'ForecastCount', 'ForecastL1', 'ForecastL2', 'ForecastMAPE', 'ForecastMASE', 'ForecastCRPS',
                                             'TestCount', 'TestL1', 'TestL2', 'TestMAPE', 'TestMASE', 'TestCRPS')) 
        # print(self.mTrPerfDetails.head(self.mTrPerfDetails.shape[0]));
        lIndicator = 'Forecast' + self.mOptions.mModelSelection_Criterion;
        lBestPerf = self.mTrPerfDetails[ lIndicator ].min();
        # allow a loss of one point (0.01 of MAPE) if complexity is reduced.
        assert(not np.isnan(lBestPerf))
        self.mTrPerfDetails.sort_values(by=[lIndicator, 'Complexity', 'Model'] ,
                                        ascending=[True, True, True],
                                        inplace=True);
        self.mTrPerfDetails = self.mTrPerfDetails.reset_index(drop=True);
        if(self.mOptions.mDebugPerformance):
            self.dump_all_model_perfs_as_json()
                
        lInterestingModels = self.mTrPerfDetails[self.mTrPerfDetails[lIndicator] <= (lBestPerf + 0.01)].reset_index(drop=True);
        lInterestingModels.sort_values(by=['Complexity'] , ascending=True, inplace=True)
        # print(self.mTransformList);
        # print(lInterestingModels.head());
        lBestName = lInterestingModels['DetailedFormula'].iloc[0];
        lBestModel = lPerfsByModel[lBestName][0][2];
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
        lPerfByCategory_df.sort_values(by=[lIndicator, 'Category'] ,
                                ascending=[True, True],
                                inplace=True);
        lPerfByCategory_df = lPerfByCategory_df.reset_index(drop=True);
                
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

def run_transform_thread(arg):
    arg.mSigDec.train(arg.mInputDS, arg.mSplit, arg.mTime, arg.mSignal, arg.mHorizon, arg.mTransformation, arg.mDecompositionType);
    return arg;

def run_finalize_training(arg):
    (lSignal , sigdecs, lOptions) = arg
    
    lModelSelector = cModelSelector_OneSignal()
    lModelSelector.mOptions = lOptions
    lModelSelector.collectPerformanceIndices_ModelSelection(lSignal, sigdecs)
    if(lOptions.mCrossValidationOptions.mMethod is not None):
        lModelSelector.perform_model_selection_cross_validation()
        
    # Prediction Intervals
    lModelSelector.mBestModel.updateAllPerfs();
    lModelSelector.mBestModel.computePredictionIntervals();
    return (lSignal, lModelSelector.mPerfsByModel, lModelSelector.mBestModel, lModelSelector.mTrPerfDetails, lModelSelector.mModelShortList)

class cSignalDecompositionTrainer:
        
    def __init__(self):
        self.mSigDecBySplitAndTransform = {};
        self.mOptions = tsopts.cSignalDecomposition_Options();
        self.mExogenousData = None;
        self.mTransformList = {}
        pass

    def define_splits(self):
        lSplits = [None]
        if(self.mOptions.mCrossValidationOptions.mMethod is not None):
            lFolds = self.mOptions.mCrossValidationOptions.mNbFolds
            lRatio = 1.0 / lFolds
            lSplits = [(k * lRatio , lRatio , 0.0) for k in range(lFolds // 2, lFolds)]
        return lSplits


    def train(self, iInputDS, iSplits, iTime, iSignals, iHorizon):
        self.train_all_transformations(iInputDS, iSplits, iTime, iSignals, iHorizon);
        self.finalize_training()
        # self.cleanup_after_model_selection()
    

    def finalize_training(self):
        args = [];
        for (lSignal , sigdecs) in self.mSigDecBySplitAndTransform.items():
            args = args + [(lSignal, sigdecs, self.mOptions)]

        self.mPerfsByModel = {}
        self.mTrPerfDetails = {}
        self.mModelShortList = {}
        self.mBestModels = {}
        NCores = min(len(args) , self.mOptions.mNbCores) 
        lTimer = tsutil.cTimer(("FINALIZE_TRAINING",
                                {"Signals" : [lSignal for (lSignal , sigdecs) in self.mSigDecBySplitAndTransform.items()],
                                 "Transformations" : [(lSignal, sorted(list(lSigDecs.keys()))) for (lSignal , lSigDecs) in self.mSigDecBySplitAndTransform.items()],
                                 "Cores" : NCores}))
        if(self.mOptions.mParallelMode and NCores > 1):
            from multiprocessing import Pool
            pool = Pool(NCores)
        
            for res in pool.imap(run_finalize_training, args):
                (lSignal, lPerfsByModel, lBestModel, lPerfDetails, lModelShortList) = res
                assert(self.mPerfsByModel.get(lSignal) is None)
                self.mPerfsByModel[lSignal] = lPerfsByModel;
                self.mBestModels[lSignal] = lBestModel
                self.mTrPerfDetails[lSignal] = lPerfDetails
                self.mModelShortList[lSignal] = lModelShortList
            pool.close()
            pool.join()
        else:
            for arg in args:
                res = run_finalize_training(arg)
                (lSignal, lPerfsByModel, lBestModel, lPerfDetails, lModelShortList) = res
                assert(self.mPerfsByModel.get(lSignal) is None)
                self.mPerfsByModel[lSignal] = lPerfsByModel;
                self.mBestModels[lSignal] = lBestModel
                self.mTrPerfDetails[lSignal] = lPerfDetails
                self.mModelShortList[lSignal] = lModelShortList
                
        
            

    def defineTransformations(self, iInputDS, iSplit, iTime, iSignal):
        lTransformationEstimator = tstransf.cTransformationEstimator()
        lTransformationEstimator.mOptions = self.mOptions
        lTransformationEstimator.defineTransformations(iInputDS, iTime, iSignal)
        self.mTransformList[(iSignal, iSplit)] = lTransformationEstimator.mTransformList
            
        
    def train_all_transformations(self , iInputDS, iSplits, iTimes, iSignals, iHorizons):
        # print([transform1.mFormula for transform1 in self.mTransformList]);
        args = [];
        for lSignal in iSignals:
            self.mSigDecBySplitAndTransform[lSignal] = {}
        lActiveDecompositionTypes = [decomp_type for (decomp_type, active_status) in self.mOptions.mActiveDecompositionTypes.items() if (active_status is True)]
        for lSplit in iSplits:
            for lSignal in iSignals:
                self.defineTransformations(iInputDS, lSplit, iTimes[lSignal], lSignal);
                for transform1 in self.mTransformList[(lSignal, lSplit)]:
                    for decomp_type in lActiveDecompositionTypes:
                        arg = cTraining_Arg(transform1.get_name(""));
                        arg.mName = (lSignal, str(lSplit) , transform1.get_name(""), decomp_type)
                        arg.mDecompositionType= decomp_type
                        arg.mSigDec = cSignalDecompositionOneTransform();
                        arg.mSigDec.mOptions = copy.copy(self.mOptions);
                        arg.mSigDec.mOptions.mCustomSplit = lSplit
                        arg.mSplit = lSplit
                        arg.mSigDec.mExogenousData = self.mExogenousData[lSignal];
                        arg.mInputDS = iInputDS;
                        arg.mTime = iTimes[lSignal];
                        arg.mSignal = lSignal;
                        arg.mHorizon = iHorizons[lSignal];
                        arg.mTransformation = transform1;
                        arg.mOptions = self.mOptions;
                        arg.mExogenousData = self.mExogenousData[lSignal];
                        arg.mResult = None;
                        args.append(arg);

        NCores = min(len(args) , self.mOptions.mNbCores)
        lTimer = tsutil.cTimer(("SIGNAL_TRAINING",{"Signals" : iSignals,
                                                   "Transformations" : [arg.mName for arg in args],
                                                   "Cores" : NCores}))
        
        if(self.mOptions.mParallelMode and NCores > 1):
            from multiprocessing import Pool
            pool = Pool(NCores)
            for res in pool.imap(run_transform_thread, args):
                lSignal = res.mName[0]
                self.mSigDecBySplitAndTransform[lSignal][res.mName] = res.mSigDec;
            pool.close()
            pool.join()
        else:
            for arg in args:
                res = run_transform_thread(arg)
                lSignal = res.mName[0]
                self.mSigDecBySplitAndTransform[lSignal][res.mName] = res.mSigDec;
            
    def cleanup_after_model_selection(self):
        lSigDecByTransform = {}
        for (lSignal , sigdecs) in self.mSigDecBySplitAndTransform.items():
            lBestTransformationName = self.mBestModels[lSignal].mTransformation.get_name("")
            for (name, sigdec) in self.mSigDecBySplitAndTransform[lSignal].items():
                if(name == lBestTransformationName):
                    for modelname in sigdec.mPerfsByModel.keys():
                        # store only model names here.
                        sigdec.mPerfsByModel[modelname][0] = modelname
                        lSigDecByTransform[lSignal][name]  = sigdec                
            # delete failing transformations
        del self.mSigDecBySplitAndTransform
        self.mSigDecBySplitAndTransform = lSigDecByTransform    

def forecast_one_signal(arg):
    (lSignal, iDecomsposition, iInputDS, iHorizon) = arg
    lBestModel = iDecomsposition.mBestModels[lSignal]
    lMissingImputer = tsmiss.cMissingDataImputer()
    lMissingImputer.mOptions = iDecomsposition.mOptions
    lInputDS = iInputDS[[lBestModel.mTime, lSignal]].copy()
    lInputDS = lMissingImputer.apply(lInputDS, lBestModel.mTime, lBestModel.mOriginalSignal)
    lForecastFrame_i = lBestModel.forecast(lInputDS, iHorizon);
    return (lBestModel.mTimeInfo, lForecastFrame_i)


class cSignalDecompositionForecaster:

    def __init__(self):
        pass

    def merge_frames(self, iFullFrame, iOneSignalFrame, iTimeInfo):
        if(iFullFrame is None):
            return iOneSignalFrame
        lTime = iFullFrame.columns[0]
        lOneSignalCommonColumns = [col for col in iOneSignalFrame.columns if col in iFullFrame.columns]
        lOneSignalCommonColumns = [col for col in lOneSignalCommonColumns if col not in [lTime , iTimeInfo.mTime]]
        lOneSignalFrame = iOneSignalFrame.drop(lOneSignalCommonColumns, axis = 1)
        lForecastFrame = iFullFrame.merge(lOneSignalFrame, how='left', left_on=lTime, right_on=iTimeInfo.mTime);
        return lForecastFrame
    
    def forecast(self, iDecomsposition, iInputDS, iHorizons):
        lHorizons = {}
        for sig in iDecomsposition.mSignals:
            if(dict == type(iHorizons)):
                lHorizons[sig] = iHorizons[sig]
            else:
                lHorizons[sig] = int(iHorizons)
        
        lForecastFrame = None
        args = [];
        for lSignal in iDecomsposition.mSignals:
            args = args + [(lSignal, iDecomsposition, iInputDS, lHorizons[lSignal])]

        NCores = min(len(args) , iDecomsposition.mOptions.mNbCores) 
        if(iDecomsposition.mOptions.mParallelMode and  NCores > 1):
            from multiprocessing import Pool
            pool = Pool(NCores)
        
            for res in pool.imap(forecast_one_signal, args):
                (lTimeInfo, lForecastFrame_i) = res
                lForecastFrame = self.merge_frames(lForecastFrame, lForecastFrame_i, lTimeInfo)
                del lForecastFrame_i
            pool.close()
            pool.join()
        else:
            for arg in args:
                res = forecast_one_signal(arg)
                (lTimeInfo, lForecastFrame_i) = res
                lForecastFrame = self.merge_frames(lForecastFrame, lForecastFrame_i, lTimeInfo)
                del lForecastFrame_i
                
        return lForecastFrame;
        
class cSignalDecomposition:
        
    def __init__(self):
        self.mSigDecBySplitAndTransform = {};
        self.mOptions = tsopts.cSignalDecomposition_Options();
        self.mExogenousData = None;
        pass

    def checkData(self, iInputDS, iTime, iSignal, iHorizon, iExogenousData):        
        if(iHorizon != int(iHorizon)):
            raise tsutil.PyAF_Error("PYAF_ERROR_NON_INTEGER_HORIZON " + str(iHorizon));
        if(iHorizon < 1):
            raise tsutil.PyAF_Error("PYAF_ERROR_NEGATIVE_OR_NULL_HORIZON " + str(iHorizon));
        if(iTime not in iInputDS.columns):
            raise tsutil.PyAF_Error("PYAF_ERROR_TIME_COLUMN_NOT_FOUND " + str(iTime));
        for lSignal in [iSignal]:
            if(lSignal not in iInputDS.columns):
                raise tsutil.PyAF_Error("PYAF_ERROR_SIGNAL_COLUMN_NOT_FOUND " + str(lSignal));
            type2 = iInputDS[lSignal].dtype
            # print(type2)
            if(type2.kind != 'i' and type2.kind != 'u' and type2.kind != 'f'):
                raise tsutil.PyAF_Error("PYAF_ERROR_SIGNAL_COLUMN_TYPE_NOT_ALLOWED '" + str(lSignal) + "' '" + str(type2) + "'");
        type1 = iInputDS[iTime].dtype
        # print(type1)
        if(type1.kind != 'M' and type1.kind != 'i' and type1.kind != 'u' and type1.kind != 'f'):
            raise tsutil.PyAF_Error("PYAF_ERROR_TIME_COLUMN_TYPE_NOT_ALLOWED '" + str(iTime) + "' '" + str(type1) + "'");
        # time in exogenous data should be the strictly same type as time in training dataset (join needed)
        if(iExogenousData is not None):
            lExogenousDataFrame = iExogenousData[0];
            lExogenousVariables = iExogenousData[1];
            if(iTime not in lExogenousDataFrame.columns):
                raise tsutil.PyAF_Error("PYAF_ERROR_TIME_COLUMN_NOT_FOUND_IN_EXOGENOUS " + str(iTime));
            for exog in lExogenousVariables:
                if(exog not in lExogenousDataFrame.columns):
                    raise tsutil.PyAF_Error("PYAF_ERROR_EXOGENOUS_VARIABLE_NOT_FOUND " + str(exog));
                
            type3 = lExogenousDataFrame[iTime].dtype
            if(type1 != type3):
                raise tsutil.PyAF_Error("PYAF_ERROR_INCOMPATIBLE_TIME_COLUMN_TYPE_IN_EXOGENOUS '" + str(iTime) + "' '" + str(type1)  + "' '" + str(type3) + "'");

    def reinterpret_by_signal_args(self, iTimes, iSignals, iHorizons, iExogenousData):
        # backward compatibility
        self.mSignals = iSignals
        if(str == type(iSignals)):
            self.mSignals = [iSignals]
        self.mDateColumns = {}
        for sig in self.mSignals:
            if(dict == type(iTimes)):
                self.mDateColumns[sig] = iTimes[sig]
            else:
                self.mDateColumns[sig] = iTimes
        self.mHorizons = {}
        for sig in self.mSignals:
            if(dict == type(iHorizons)):
                self.mHorizons[sig] = iHorizons[sig]
            else:
                self.mHorizons[sig] = int(iHorizons)
        self.mExogenousData = {}
        for sig in self.mSignals:
            if(dict == type(iExogenousData)):
                self.mExogenousData[sig] = iExogenousData[sig]
            else:
                self.mExogenousData[sig] = iExogenousData
        
            
    def train(self , iInputDS, iTimes, iSignals, iHorizons, iExogenousData = None):
        self.reinterpret_by_signal_args(iTimes, iSignals, iHorizons, iExogenousData)
        # print(iInputDS.shape, iInputDS.columns, self.mSignals, self.mDateColumns, self.mHorizons)

        for sig in self.mSignals:
            self.checkData(iInputDS, self.mDateColumns[sig], sig, self.mHorizons[sig], self.mExogenousData[sig]);

        self.mTrainingDataset = iInputDS; 

        lTrainer = cSignalDecompositionTrainer()
        lSplits = lTrainer.define_splits()
        lTrainer.mOptions = self.mOptions;
        lTrainer.mExogenousData = self.mExogenousData;
        
        lTrainer.train(iInputDS, lSplits , self.mDateColumns, self.mSignals, self.mHorizons)
        self.mBestModels = lTrainer.mBestModels
        self.mTrPerfDetailsBySignal = lTrainer.mTrPerfDetails
        self.mModelShortListBySignal = lTrainer.mModelShortList
        # some backward compatibility
        lFirstSignal = self.mSignals[0] 
        self.mTrPerfDetails = lTrainer.mTrPerfDetails[lFirstSignal]
        self.mModelShortList = lTrainer.mModelShortList[lFirstSignal]
        self.mBestModel = self.mBestModels[lFirstSignal]


    def forecast(self , iInputDS, iHorizon):
        lTimer = tsutil.cTimer(("FORECASTING", {"Signals" : self.mSignals, "Horizon" : iHorizon}))
        lForecaster = cSignalDecompositionForecaster()
        lInputDS = sample_signal_if_needed(iInputDS, self.mOptions)
        lForecastFrame = lForecaster.forecast(self, lInputDS, iHorizon)
        return lForecastFrame;


    def getModelFormula(self):
        lFormula = {}
        for lSignal in self.mSignals:
            lFormula[lSignal] = self.mBestModel.getFormula();
        return lFormula;


    def getModelInfo(self):
        for lSignal in self.mSignals:
            self.mBestModels[lSignal].getInfo()

    def to_dict(self, iWithOptions = False):
        dict1 = {}
        for lSignal in self.mSignals:
            dict1[lSignal] = self.mBestModels[lSignal].to_dict(iWithOptions);
        return dict1
        
    def standardPlots(self, name = None, format = 'png'):
        lTimer = tsutil.cTimer(("PLOTTING", {"Signals" : self.mSignals}))
        for lSignal in self.mSignals:
            lName = name
            if(name is not None):
                lName = str(name) + "_" + str(lSignal)
            self.mBestModels[lSignal].standardPlots(lName, format);
        
    def getPlotsAsDict(self):
        lDict = {}
        for lSignal in self.mSignals:
            lDict[lSignal] = self.mBestModels[lSignal].getPlotsAsDict();
        return lDict;
