# Copyright (C) 2023 Antoine Carme <Antoine.Carme@outlook.com>
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
from . import ModelSelection_Voting as tsvote
from . import ModelSelection_Legacy as tsleg


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
        # lTimer = tsutil.cTimer(("UPDATE_PERFS_FOR_ALL_MODELS", {"Signal" : self.mOriginalSignal, "Transformation" : self.mSignal, "DecompositionType" : self.mDecompositionType}, len(iModels)))
        self.mPerfsByModel = {}
        for model in iModels.keys():
            # lTimer2 = tsutil.cTimer(("UPDATE_PERFS_FOR_MODEL", model))
            iModels[model].updatePerfs();
            
        for (name, model) in iModels.items():
            # print(name, model.__dict__);
            lComplexity = model.getComplexity_as_ordering_string();
            (lFitPerf, lForecastPerf, lTestPerf) = model.get_aggregated_criterion_values_for_model_selection();
            lSplit = model.mTimeInfo.mOptions.mCustomSplit
            
            self.mPerfsByModel[(self.mSignal, self.mDecompositionType, lSplit, model.mOutName)] = [(self.mSignal, self.mDecompositionType, model), lComplexity, lFitPerf , lForecastPerf, lTestPerf];
            
        return iModels;


    def train(self , iInputDS, iSplit, iTime, iSignal,
              iHorizon, iTransformation, iDecomspositionType):
        lTimer = tsutil.cTimer(("TRAINING", {"Signal" : iSignal,
                                             "Horizon" : iHorizon,
                                             "Transformation" : iTransformation.get_name(iSignal),
                                             "DecompositionType" : iDecomspositionType}))
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



def run_transform_thread(arg):
    arg.mSigDec.train(arg.mInputDS, arg.mSplit, arg.mTime, arg.mSignal, arg.mHorizon, arg.mTransformation, arg.mDecompositionType);
    return arg;

def run_finalize_training(arg):
    (lSignal , sigdecs, lOptions) = arg
    lModelSelector = None
    if(lOptions.mVotingMethod is None):
        lModelSelector = tsleg.create_model_selector()
    else:
        lModelSelector = tsvote.create_model_selector(lOptions.mVotingMethod)
        
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
