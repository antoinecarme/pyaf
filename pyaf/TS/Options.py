# Copyright (C) 2016 Antoine Carme <Antoine.Carme@outlook.com>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np


class cModelControl:

    gKnownDecompositionTypes = ['T+S+R', 'TS+R', 'TSR']
    gKnownTransformations = ['None', 'Difference', 'RelativeDifference',
                             'Integration', 'BoxCox',
                             'Quantization', 'Logit',
                             'Fisher', 'Anscombe'];
    gKnownTrends = ['ConstantTrend', 
                    'Lag1Trend', 'LinearTrend', 'PolyTrend', 
                    'MovingAverage', 'MovingMedian'];
    gKnownPeriodics = ['NoCycle', 'BestCycle',
                       'Seasonal_MonthOfYear' ,
                       'Seasonal_Second' ,
                       'Seasonal_Minute' ,
                       'Seasonal_Hour' ,
                       'Seasonal_HourOfWeek' ,
                       'Seasonal_TwoHourOfWeek' ,
                       'Seasonal_ThreeHourOfWeek' ,
                       'Seasonal_FourHourOfWeek' ,
                       'Seasonal_SixHourOfWeek' ,
                       'Seasonal_EightHourOfWeek' ,
                       'Seasonal_TwelveHourOfWeek' ,
                       'Seasonal_DayOfWeek' ,
                       'Seasonal_DayOfMonth',
                       'Seasonal_DayOfYear',
                       'Seasonal_WeekOfMonth',
                       'Seasonal_DayOfNthWeekOfMonth',
                       'Seasonal_WeekOfYear'];

    # "AutoRegression" becomes a little bit confusing as croston does not use lags (???)
    # rather use wikipedia terminology :  https://en.wikipedia.org/wiki/Decomposition_of_time_series
    # AutoRegression => "irregular component"
    gKnownAutoRegressions = ['NoAR' ,
                             'AR' , 'ARX' ,
                             'SVR', 'SVRX',
                             'MLP' , 'MLPX' ,
                             'LSTM' , 'LSTMX' ,
                             'XGB' , 'XGBX' ,
                             'CROSTON', # No CROSTONX for the moment
                             'LGB', 'LGBX'];

    def __init__(self):
        self.mActiveTransformations = {};
        self.mActivePeriodics = {};
        self.mActiveTrends = {};
        self.mActiveAutoRegressions = {};
        #  Add Multiplicative Models/Seasonals #178 
        self.mActiveDecompositionTypes = {}

        # now , set he default models
        self.set_active_transformations(cModelControl.gKnownTransformations[0:4]);
        self.set_active_trends(cModelControl.gKnownTrends[0:4]);
        self.set_active_periodics(cModelControl.gKnownPeriodics);
        self.set_active_autoregressions(cModelControl.gKnownAutoRegressions[0:3]);
        # Add Multiplicative Models/Seasonals #178.
        # Only additive models are activated by default        
        self.set_active_decomposition_types(['T+S+R']);

    def check_model_type_validity(self, category, allowed_values, value):
        from . import Utils as tsutil
        
        if(value not in allowed_values):
            raise tsutil.PyAF_Error("INVALID_MODEL_TYPE Invalid '" + category + "' Type '" + value + "'. Allowed '" + category + "' Values : " + str(allowed_values));
        
        
    def set_active_decomposition_types(self, iDecompTypes):
        for dec_type in iDecompTypes:
            self.check_model_type_validity('DecompositionType', cModelControl.gKnownDecompositionTypes, dec_type)
            
        self.mActiveDecompositionTypes = {};
        for decomp_type in cModelControl.gKnownDecompositionTypes:
            if(decomp_type in iDecompTypes):
                self.mActiveDecompositionTypes[decomp_type] = True;
            else:
                self.mActiveDecompositionTypes[decomp_type] = False;
        if(True not in self.mActiveDecompositionTypes.values()):
            # default
            self.mActiveTransformations['T+S+R'] = True;
            
    def set_active_transformations(self, transformations):
        for transformation in transformations:
            self.check_model_type_validity('Transformation', cModelControl.gKnownTransformations, transformation)
            
        self.mActiveTransformations = {};
        for transformation in cModelControl.gKnownTransformations:
            if(transformation in transformations):
                self.mActiveTransformations[transformation] = True;
            else:
                self.mActiveTransformations[transformation] = False;
        if(True not in self.mActiveTransformations.values()):
            # default
            self.mActiveTransformations['None'] = True;
    
    def set_active_trends(self, trends):
        for trend in trends:
            self.check_model_type_validity('Trend', cModelControl.gKnownTrends, trend)
            
        self.mActiveTrends = {};
        for trend in cModelControl.gKnownTrends:
            if(trend in trends):
                self.mActiveTrends[trend] = True;
            else:
                self.mActiveTrends[trend] = False;
        if(True not in self.mActiveTrends.values()):
            # default
            self.mActiveTrends['ConstantTrend'] = True;                
    
    def set_active_periodics(self, periodics):
        for period in periodics:
            self.check_model_type_validity('Periodic', cModelControl.gKnownPeriodics, period)
            
        self.mActivePeriodics = {};
        for period in cModelControl.gKnownPeriodics:
            if(period in periodics):
                self.mActivePeriodics[period] = True;
            else:
                self.mActivePeriodics[period] = False;
        if(True not in self.mActivePeriodics.values()):
            # default
            self.mActivePeriodics['NoCycle'] = True;
                    
    def set_active_autoregressions(self, autoregs):
        for autoreg in autoregs:
            self.check_model_type_validity('AutoRegression', cModelControl.gKnownAutoRegressions, autoreg)
            
        self.mActiveAutoRegressions = {};
        for autoreg in cModelControl.gKnownAutoRegressions:
            if(autoreg in autoregs):
                self.mActiveAutoRegressions[autoreg] = True;
            else:
                self.mActiveAutoRegressions[autoreg] = False;                
        if(True not in self.mActiveAutoRegressions.values()):
            # default
            self.mActiveAutoRegressions['NoAR'] = True;

    def disable_all_transformations(self):
        self.set_active_transformations([]);
    
    def disable_all_trends(self):
        self.set_active_trends([]);
    
    def disable_all_periodics(self):
        self.set_active_periodics([]);
    
    def disable_all_autoregressions(self):
        self.set_active_autoregressions([]);
    
class cCrossValidationOptions:
    def __init__(self):
        self.mMethod = None;
        self.mNbFolds = 10

class cCrostonOptions:
    def __init__(self):
        # can be : "CROSTON" , "SBJ" , "SBA"
        self.mMethod = None;
        # alpha value or None to use optimal alpha based on RMSE
        self.mAlpha = 0.1
        # use "L2" by default, MAPE is not suitable (a lot of zeros in the signal) ?
        self.mAlphaCriterion = "L2"
        # minimum amount/percentage of zeros for a series to be intermittent
        self.mZeroRate = 0.1

class cMissingDataOptions:

    def __init__(self):
        self.mSignalMissingDataImputation = None  # [None , "DiscardRow", "Interpolate", "Mean", "Median" , "Constant" , "PreviousValue"]
        self.mTimeMissingDataImputation = None  # [None , "DiscardRow", "Interpolate"]
        self.mConstant = 0.0
        
class cSignalDecomposition_Options(cModelControl):
    
    def __init__(self):
        super().__init__();
        self.mParallelMode = True;
        self.mNbCores = 8;
        self.mSeed = 1960
        self.mEstimRatio = 0.8; # to be deprecated when cross validation is OK.
        self.mCustomSplit = None
        self.mAddPredictionIntervals = True
        self.mActivateSampling = True # sampling can be used for very large time series
        self.mSamplingThreshold = 8192 # Time series larger than this threshold will be sampled.
        self.enable_fast_mode();
        self.mTimeDeltaComputationMethod = "AVG"; # can be "AVG", "MODE", "USER"
        self.mUserTimeDelta = None;
        self.mBusinessDaysOnly = False;
        self.mMaxExogenousCategories = 5;
        self.mNoBoxCoxOrders = [];
        self.mBoxCoxOrders = [-2.0, -1.0 , 0.0,  2.0];
        self.mExtensiveBoxCoxOrders = [-2, -1, -0.5, -0.33 , -0.25 , 0.0, 2, 0.5, 0.33 , 0.25];
        self.mMaxFeatureForAutoreg = 1000;
        self.mModelSelection_Criterion = "MASE";
        self.mCycle_Criterion = "MASE";
        self.mCycle_Criterion_Threshold = None;
        self.mCycle_Encoding_Scheme = "Target_Median"; # "Target_Mean" or "Target_Median"
        self.mHierarchicalCombinationMethod = "BU";
        self.mForecastRectifier = None # can be "relu" to force positive forecast values
        self.mXGBOptions = None
        self.mLGBMOptions = None
        self.mCrossValidationOptions = cCrossValidationOptions()
        self.mCrostonOptions = cCrostonOptions()
        self.mMissingDataOptions = cMissingDataOptions()
        self.mDL_Backends = ("PyTorch", ) # Pytorch is the only supported backend for now
        self.mPytorch_Options = None
        self.mKeras_Options = None
        self.mVotingMethod = "Condorcet" # Or None for Legacy Method (backward compatibility with PyAF 4.0). 
        self.disableDebuggingOptions();

    def disableDebuggingOptions(self):
        self.mDebug = False;
        self.mDebugCycles = False;
        self.mDebugAR = False;
        self.mDebugProfile = False;
        self.mDebugPerformance = False;
        
        
    def enable_slow_mode(self):
        self.mQuantiles = [5, 10, 20]; # quintiles, deciles, and vingtiles;)
        self.mMovingAverageLengths = [5, 7, 12, 24 , 30, 60];
        self.mMovingMedianLengths = [5, 7, 12, 24 , 30, 60];
        # PyAF does not detect complex seasonal patterns #73.
        # use unlimited cycle lengths in slow mode
        self.mCycleLengths = None;

        self.set_active_transformations(cModelControl.gKnownTransformations);
        self.set_active_trends(cModelControl.gKnownTrends);
        self.set_active_periodics(cModelControl.gKnownPeriodics);
        self.set_active_autoregressions(cModelControl.gKnownAutoRegressions);
        self.set_active_decomposition_types(cModelControl.gKnownDecompositionTypes);
        
        self.mMaxAROrder = 64;
        self.mFilterSeasonals = False
        # disable cross validation
        # self.mCrossValidationOptions.mMethod = "TSCV";
        self.mActivateSampling = False

    def enable_fast_mode(self):
        self.mQuantiles = [5, 10, 20]; # quintiles, deciles, and vingtiles;)
        self.mMovingAverageLengths = [5, 7, 12, 24 , 30, 60];
        self.mMovingMedianLengths = [5, 7, 12, 24 , 30, 60];
        
        self.mCycleLengths = [5, 7, 12, 24 , 30, 60];

        self.mMaxAROrder = 64;
        self.mFilterSeasonals = True


    # Add a low-memory mode for Heroku #25
    def enable_low_memory_mode(self):
        self.mMaxAROrder = 7;
        self.set_active_transformations(['None']);
        self.mParallelMode = False;
        self.mFilterSeasonals = True
        
    def has_module_installed(self, module_name):
        import importlib
        spec = importlib.util.find_spec(module_name)
        return spec is not None

    def get_available_DL_Backend(self):
        # pick the first available backend
        for lBackend in self.mDL_Backends:
            if(lBackend == "PyTorch"):
                if(self.has_module_installed("torch")):
                    return lBackend
            if(lBackend == "Keras"):
                if(self.has_module_installed("tensorflow")):
                    return lBackend
        return None        

    def getPytorchOrKerasClass(self, iModel):
        lBackend = self.get_available_DL_Backend()
        if(lBackend == "PyTorch"):
            from . import Pytorch_Models as tspytorch
            lDict = {"LSTM" : tspytorch.cLSTM_Model, "MLP" : tspytorch.cMLP_Model}
            return lDict.get(iModel)
        if(lBackend == "Keras"):
            from . import Keras_Models as tskeras
            lDict = {"LSTM" : tskeras.cLSTM_Model, "MLP" : tskeras.cMLP_Model}
            return lDict.get(iModel)
        return None
    
    def hasPytorchOrKerasInstalled(self, iModel):
        return self.has_module_installed('torch') or self.has_module_installed('tensorflow')

    def  canBuildXGBoostModel(self, iModel):
        return self.has_module_installed('xgboost')

    def  canBuildLightGBMModel(self, iModel):
        return self.has_module_installed('lightgbm')
