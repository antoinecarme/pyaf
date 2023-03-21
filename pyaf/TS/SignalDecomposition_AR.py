# Copyright (C) 2016 Antoine Carme <Antoine.Carme@outlook.com>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np


from . import Time as tsti
from . import Perf as tsperf
from . import Plots as tsplot
from . import Utils as tsutil
from . import Complexity as tscomplex

class cAbstractAR:
    def __init__(self , cycle_residue_name, iExogenousInfo = None):
        self.mTimeInfo = tsti.cTimeInfo()
        self.mCycleFrame = None
        self.mARFrame = None        
        self.mCycleResidueName = cycle_residue_name
        self.mCycle = None
        self.mTrend = None
        self.mComplexity = tscomplex.eModelComplexity.High;
        self.mFormula = None;
        self.mTargetName = self.mCycleResidueName;
        self.mInputNames = [];
        self.mExogenousInfo = iExogenousInfo;
        self.mLagsForSeries = {cycle_residue_name : []}

    def compute_ar_residue(self, df):
        target = df[self.mCycleResidueName].values
        lSignal = df[self.mSignal].values
        lTrend = df[self.mTrend.mOutName].values
        lCycle = df[self.mCycle.mOutName].values
        lAR = df[self.mOutName].values
        if(self.mDecompositionType in ['T+S+R']):
            df[self.mOutName + '_residue'] = lSignal - lTrend - lCycle - lAR
        if(self.mDecompositionType in ['TS+R']):
            df[self.mOutName + '_residue'] = lSignal - lTrend * lCycle - lAR
        else:
            df[self.mOutName + '_residue'] = lSignal - (lTrend * lCycle * lAR)
        # df_detail = df[[self.mSignal, self.mTrend.mOutName, self.mCycle.mOutName, self.mOutName, self.mOutName + '_residue']]
        # print("compute_ar_residue_detail ", (self.mOutName, self.mDecompositionType, df_detail.describe(include='all').to_dict()))
        
    def plot(self):
        tsplot.decomp_plot(self.mARFrame, self.mTimeInfo.mNormalizedTimeColumn,
                           self.mCycleResidueName, self.mOutName , self.mOutName + '_residue', horizon = self.mTimeInfo.mHorizon);


    def register_lag(self, series, p):
        name = series+'_Lag' + str(p);
        # print("register_lag", (series , p , name))
        assert(name not in self.mInputNames)
        self.mInputNames.append(name);
        self.mLagsForSeries[series] = self.mLagsForSeries.get(series , [])
        self.mLagsForSeries[series].append(p)
        
    def dumpCoefficients(self):
        pass
    
    def computePerf(self):
        self.mARFitPerf= tsperf.cPerf();
        self.mARForecastPerf= tsperf.cPerf();
        (lFrameFit, lFrameForecast, lFrameTest) = self.mSplit.cutFrame(self.mARFrame);
        self.mARFitPerf.computeCriterionValues(
            lFrameFit[self.mCycleResidueName],
            lFrameFit[self.mOutName],
            [self.mTimeInfo.mOptions.mModelSelection_Criterion],
            self.mOutName)
        self.mARForecastPerf.computeCriterionValues(
            lFrameForecast[self.mCycleResidueName],
            lFrameForecast[self.mOutName],
            [self.mTimeInfo.mOptions.mModelSelection_Criterion],
            self.mOutName)

    def shift_series(self, series, p, idefault):
        N = series.shape[0];
        new_values = np.append([ idefault ]*p, series[0:N-p])
        return new_values
    
    def getDefaultValue(self, series):
        return self.mDefaultValues[series];

    def generateLagsForForecast(self, df):
        lDict = {}
        # lDict[self.mCycleResidueName] = df[self.mCycleResidueName]
        series = self.mCycleResidueName
        lSeries = df[self.mCycleResidueName]
        #  Investigate Large Horizon Models #213 : The model can produce overflows in its inputs when iterated. 
        lSeries = lSeries.values.clip(-1e+10, +1e10)
        for p in self.mLagsForSeries[self.mCycleResidueName]:
            name = series +'_Lag' + str(p);
            lShiftedSeries = self.shift_series(lSeries, p , self.mDefaultValues[series]); 
            lDict[name] = lShiftedSeries
        # Exogenous variables lags
        if(self.mExogenousInfo is not None):
            for ex in self.mExogenousInfo.mEncodedExogenous:
                if(self.mLagsForSeries.get(ex)):
                    for p in self.mLagsForSeries[ex]:
                        name = ex +'_Lag' + str(p);
                        lShiftedSeries = self.shift_series(df[ex], p , self.mDefaultValues[ex]); 
                        lDict[name] = lShiftedSeries
        cols = lDict.keys()
        lArray = np.concatenate([lDict[k].reshape(-1, 1) for k in lDict.keys()], axis = 1, dtype = lSeries.dtype)
        lag_df = pd.DataFrame(data = lArray, columns= cols, index = df.index, dtype = lSeries.dtype)
        return lag_df;


class cZeroAR(cAbstractAR):
    def __init__(self , cycle_residue_name):
        super().__init__(cycle_residue_name, None)
        self.mOutName = self.mCycleResidueName +  '_NoAR'
        self.mNbLags = 0;
        self.mFormula = "NoAR";
        self.mComplexity = tscomplex.eModelComplexity.Low;
        self.mConstantValue = 0.0
        
    def fit(self):
        series = self.mCycleResidueName; 
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mConstantValue = 0.0
        if(self.mDecompositionType in ['TSR']):
            # multiplicative models
            self.mConstantValue = 1.0
        
        # self.mTimeInfo.addVars(self.mARFrame);
        # self.mARFrame[series] = self.mCycleFrame[series]
        self.mARFrame[self.mOutName] = self.mConstantValue;
        self.mARFrame[self.mCycle.mOutName] = self.mConstantValue;
        self.mARFrame[self.mTrend.mOutName] = self.mConstantValue;
        self.compute_ar_residue(self.mARFrame)
        assert(self.mARFrame.shape[0] > 0)
                

    def transformDataset(self, df, horizon_index = 1):
        df[self.mOutName] = self.mConstantValue;
        self.compute_ar_residue(df)
        assert(df.shape[0] > 0)
        return df;



class cAutoRegressiveEstimator:
    def __init__(self):
        self.mTimeInfo = tsti.cTimeInfo()
        self.mCycleFrame = None
        self.mARFrame = None
        self.mARList = {}
        self.mExogenousInfo = None;
        
    def plotAR(self):
        for trend in self.mTrendList:
            for cycle in self.mCycleList[trend]:
                cycle_residue = cycle.getCycleResidueName();
                for autoreg in self.mARList[cycle_residue]:
                    autoreg.plot(); 

    def is_not_constant(self, iSeries):
        lFirst = iSeries[0];
        for lValue in iSeries[1:]:
            if(lValue != lFirst):
                return True;
        return False;

    def shift_series(self, series, p):
        N = series.shape[0];
        new_values = np.append([ series[0] ]*p, series[0:N-p])
        return new_values

    def generateLagsForTraining(self, df, series, pMinMax):
        (pmin, pmax) = pMinMax
        lSeries = df[series];
        self.mDefaultValues[series] = lSeries.values[0];
        lDict = {}
        lags = []
        for p in range(pmin, pmax+1):
            name = series+'_Lag' + str(p)
            lShiftedSeries = self.shift_series(lSeries.values, p)
            lShiftedEstim = self.mSplit.getEstimPart(lShiftedSeries);
            lAcceptable = self.is_not_constant(lShiftedEstim);
            if(lAcceptable):
                lDict[name] = lShiftedSeries
                lags.append((series, p))
        lag_df = pd.DataFrame(lDict, index = df.index, dtype = lSeries.dtype)
        return (lag_df, lags)

    def preselect_exog_vars(self, df, cycle_residue):
        P = self.get_nb_lags();
        lMaxFeatures = self.mOptions.mMaxFeatureForAutoreg;
        lExogCount = len(self.mExogenousInfo.mEncodedExogenous);
        lNbVars = P * lExogCount;
        if(lNbVars <= lMaxFeatures):
            return self.mExogenousInfo.mEncodedExogenous
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_regression
        lPreselectionFeatureSelector =  SelectKBest(f_regression, k= max(1, lMaxFeatures // P))
        df_Estim = self.mSplit.getEstimPart(df)
        lARInputs = df_Estim[self.mExogenousInfo.mEncodedExogenous].values
        lARTarget = df_Estim[cycle_residue].values
        lPreselectionFeatureSelector.fit(lARInputs, lARTarget)
        lSupport = lPreselectionFeatureSelector.get_support(indices=True);
        lPreselected = [self.mExogenousInfo.mEncodedExogenous[k] for k in lSupport];
        return lPreselected

    def addLagsForTraining(self, df, cycle_residue):
        P = self.get_nb_lags();
        lag_df, lags = self.generateLagsForTraining(df, cycle_residue, (1, P));
        lag_dfs = [lag_df]
        for autoreg in self.mARList[cycle_residue]:
            for lag in lags:
                (name , p) = lag
                autoreg.register_lag(name, p)

        # Exogenous variables lags
        lUseExog = False # Exog variables can be configured but not used ("AR" activated and "ARX" disabled).
        for autoreg in self.mARList[cycle_residue]:
            if(autoreg.mExogenousInfo is not None): # ARX,XGBX, ... only
                lUseExog = True
        if(lUseExog):
            P1 = P;
            autoreg.mNbExogenousLags = P1;
            lEncodedExogenous = self.preselect_exog_vars(df, cycle_residue)
            for ex in lEncodedExogenous:
                (lag_df, lags_ex) = self.generateLagsForTraining(df, ex, (1, P1));
                lag_dfs = lag_dfs + [lag_df]        
                for autoreg in self.mARList[cycle_residue]:
                    if(autoreg.mExogenousInfo is not None): # ARX,XGBX, ... only
                        for lag in lags_ex:
                            (name , p) = lag
                            autoreg.register_lag(name, p)

        self.mARFrame = pd.concat([self.mARFrame] + lag_dfs, axis = 1)


    # @profile
    def estimate_ar_models_for_cycle(self, cycle_residue):
        logger = tsutil.get_pyaf_logger();
        self.mARFrame = pd.DataFrame(index = self.mCycleFrame.index);
        self.mTimeInfo.addVars(self.mARFrame);
        self.mCycleFrame[cycle_residue] = self.mCycleFrame[cycle_residue]            
        self.mARFrame[cycle_residue] = self.mCycleFrame[cycle_residue]            

        self.mDefaultValues = {};

        self.addLagsForTraining(self.mCycleFrame, cycle_residue);


        # print(self.mARFrame.info());

        lCleanListOfArModels = [];
        for autoreg in self.mARList[cycle_residue]:
            self.mARFrame[autoreg.mTrend.mOutName] = autoreg.mCycle.mTrendFrame[autoreg.mTrend.mOutName]            
            self.mARFrame[autoreg.mCycle.mOutName] = self.mCycleFrame[autoreg.mCycle.mOutName]            
            if((autoreg.mFormula == "NoAR") or (len(autoreg.mInputNames) > 0)):
                lCleanListOfArModels.append(autoreg);
            else:
                if(self.mOptions.mDebugAR):
                    logger.info("SKIPPING_AR_MODEL_NO_VALID_INPUTS " + autoreg.mOutName);
                
        if(len(lCleanListOfArModels) == 0):
            lZeroAR = cZeroAR(cycle_residue)
            cycle = self.mARList[cycle_residue][0].mCycle
            lZeroAR.mCycle = cycle
            lZeroAR.mTrend = cycle.mTrend
            lCleanListOfArModels = [ lZeroAR ]
            
        self.mARList[cycle_residue] = lCleanListOfArModels;
        
        for autoreg in self.mARList[cycle_residue]:
            autoreg.mOptions = self.mOptions;
            autoreg.mCycleFrame = self.mCycleFrame;
            autoreg.mARFrame = self.mARFrame
            autoreg.mTimeInfo = self.mTimeInfo;
            autoreg.mSplit = self.mSplit;
            autoreg.mDefaultValues = self.mDefaultValues;
            autoreg.mDecompositionType = self.mDecompositionType
            lTimer = None
            if(self.mOptions.mDebugAR):
                lTimer = tsutil.cTimer(("TRAINING_AR_MODEL", autoreg.mFormula, autoreg.mCycleResidueName))
            autoreg.fit();
            if(self.mOptions.mDebugAR):
                autoreg.computePerf();

    def check_not_nan(self, sig , name):
        #print("check_not_nan");
        if(np.isnan(sig[:-1]).any()):
            logger = tsutil.get_pyaf_logger();
            logger.error("CYCLE_RESIDUE_WITH_NAN_IN_SIGNAL" + str(sig));
            raise tsutil.Internal_PyAF_Error("INVALID_COLUMN _FOR_CYCLE_RESIDUE ['"  + name + "'");
        pass


    def get_nb_lags(self):
        lLags = self.mCycleFrame.shape[0] // 4;
        if(lLags >= self.mOptions.mMaxAROrder):
            lLags = self.mOptions.mMaxAROrder;
        return lLags


    def add_model_if_activated(self, cycle_residue, model_type_str, model_class, iLags, iAddExogenous = True):
        if(self.mOptions.mActiveAutoRegressions[model_type_str]):
            lNewModel = model_class(cycle_residue, iLags);
            self.mARList[cycle_residue] = self.mARList[cycle_residue] + [lNewModel];
        if(iAddExogenous and (self.mExogenousInfo is not None) and self.mOptions.mActiveAutoRegressions[model_type_str + 'X']):
            lNewModelX = model_class(cycle_residue, iLags, self.mExogenousInfo);
            self.mARList[cycle_residue] = self.mARList[cycle_residue] + [lNewModelX];
                
        
    # @profile
    def estimate(self):
        from . import Scikit_Models as tsscikit

        lTimer = None
        if(self.mOptions.mDebugAR):
            lTimer = tsutil.cTimer(("TRAINING_AR_MODELS", {"Signal" : self.mTimeInfo.mSignal}))
            
        logger = tsutil.get_pyaf_logger();
        self.mSkippedARList = []
        lNeedExogenous = False;

        for trend in self.mTrendList:
            for cycle in self.mCycleList[trend]:
                cycle_residue = cycle.getCycleResidueName();
                if(self.mOptions.mDebugAR):
                    self.check_not_nan(self.mCycleFrame[cycle_residue], cycle_residue)
                self.mARList[cycle_residue] = [];
                if(self.mOptions.mActiveAutoRegressions['NoAR']):
                    self.mARList[cycle_residue] = [ cZeroAR(cycle_residue)];
                lLags = self.get_nb_lags()
                lThreshold = 0.001 # The signal is scaled to be between 0 and 1
                lEstimResidue = self.mSplit.getEstimPart(self.mCycleFrame[cycle_residue])
                lCycleRange = lEstimResidue.max() - lEstimResidue.min() 
                lKeep = (lEstimResidue.shape[0] > 12) and (lCycleRange >= lThreshold) # Keep this test as simple as possible.
                if(not lKeep):
                    self.mSkippedARList = self.mSkippedARList + [cycle_residue]
                    
                if(lKeep):
                    self.add_model_if_activated(cycle_residue, 'AR', tsscikit.cAutoRegressiveModel, lLags, True)
                    self.add_model_if_activated(cycle_residue, 'SVR', tsscikit.cSVR_Model, lLags, True)
                    if(self.mOptions.mActiveAutoRegressions['LSTM'] or self.mOptions.mActiveAutoRegressions['LSTMX']):
                        lLSTMClass = self.mOptions.getPytorchOrKerasClass('LSTM')
                        if(lLSTMClass is not None):
                            self.add_model_if_activated(cycle_residue, 'LSTM', lLSTMClass, lLags, True)
                    if(self.mOptions.mActiveAutoRegressions['MLP'] or self.mOptions.mActiveAutoRegressions['MLPX']):
                        lMLPClass = self.mOptions.getPytorchOrKerasClass('MLP')
                        if(lMLPClass is not None):                    
                            self.add_model_if_activated(cycle_residue, 'MLP', lMLPClass, lLags, True)
                    if(self.mOptions.mActiveAutoRegressions['XGB'] or self.mOptions.mActiveAutoRegressions['XGBX']):
                        if(self.mOptions.canBuildXGBoostModel('XGB')):
                            self.add_model_if_activated(cycle_residue, 'XGB', tsscikit.cXGBoost_Model, lLags, True)
                    if(self.mOptions.mActiveAutoRegressions['LGB'] or self.mOptions.mActiveAutoRegressions['LGBX']):
                        if(self.mOptions.canBuildLightGBMModel('LGB')):
                            self.add_model_if_activated(cycle_residue, 'LGB', tsscikit.cLightGBM_Model, lLags, True)
                    if(self.mOptions.mActiveAutoRegressions['CROSTON']):
                        from . import Intermittent_Models as interm
                        lIsSignalIntermittent = interm.is_signal_intermittent(self.mCycleFrame[cycle_residue] , self.mOptions)
                        if(lIsSignalIntermittent):
                            # TODO : need to define/design how to deal with exogenous variables in croston-based models.
                            self.add_model_if_activated(cycle_residue, 'CROSTON', interm.cCroston_Model, lLags, False)
                        
                if(len(self.mARList[cycle_residue]) == 0):
                    self.mARList[cycle_residue] = [ cZeroAR(cycle_residue)];
                for lAR in self.mARList[cycle_residue]:
                    lAR.mCycle = cycle
                    lAR.mTrend = cycle.mTrend
                    if(lAR.mExogenousInfo is not None):
                        lNeedExogenous = True

        if(len(self.mSkippedARList) > 0):
            lTenFirst = sorted(self.mSkippedARList[:10])
            lSkipInfo = (self.mDecompositionType, len(self.mSkippedARList), 10, lTenFirst)
            if(self.mOptions.mDebugAR):
                logger.info("SKIPPING_AR_MODELS_WITH_LOW_VARIANCE_FOR_CYCLE_RESIDUES " + str(lSkipInfo));
                        
        if(lNeedExogenous):
            if(self.mOptions.mDebugAR):
                logger.info("AR_MODEL_ADD_EXOGENOUS " + str(self.mCycleFrame.shape[0]) +
                      " " + str(len(self.mExogenousInfo.mEncodedExogenous)));
            self.mCycleFrame = self.mExogenousInfo.transformDataset(self.mCycleFrame);
        
        for cycle_residue in self.mARList.keys():
            self.estimate_ar_models_for_cycle(cycle_residue);
            for autoreg in self.mARList[cycle_residue]:
                autoreg.mARFrame = pd.DataFrame(index = self.mCycleFrame.index);
            del autoreg.mARFrame
        del self.mARFrame;
