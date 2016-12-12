# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np

class cSignalDecomposition_Options:
    
    def __init__(self):
        self.mParallelMode = True;
        self.mNbCores = 8;
        self.mEnablePlots = False;
        self.mEstimRatio = 0.8;
        self.mActiveTransformation = {};
        self.enable_fast_mode();
        self.mTimeDeltaComputationMethod = "AVG"; # can be "AVG", "MODE", "USER"
        self.mUserTimeDelta = None;
        self.mBusinessDaysOnly = False;
        self.mMaxExogenousCategories = 5;
        self.mNoBoxCoxOrders = [];
        self.mBoxCoxOrders = [-2.0, -1.0 , 0.0,  2.0];
        self.mExtensiveBoxCoxOrders = [-2, -1, -0.5, -0.33 , -0.25 , 0.0, 2, 0.5, 0.33 , 0.25];
        self.mEnableTrends = True;
        self.mEnableCycles = True;
        self.mEnableARModels = True;
        self.mEnableARXModels = True;
        self.mMaxFeatureForAutoreg = 1000;
        self.mModelSelection_Criterion = "L2";
        self.mCycle_Criterion = "L2";
        self.mCycle_Criterion_Threshold = None;
        self.mHierarchicalCombinationMethod = "BU";
        self.disableDebuggingOptions();

    def disableDebuggingOptions(self):
        self.mDebug = False;
        self.mDebugCycles = False;
        self.mDebugProfile = False;
        self.mDebugPerformance = False;
        
        
    def enable_slow_mode(self):
        self.mQuantiles = [5, 10, 20]; # quintiles, deciles, and vingtiles;)
        self.mEnableMovingAverageTrends = True;
        self.mEnableMovingMedianTrends = True;
        self.mEnableTimeBasedTrends = True;
        self.mMovingAverageLengths = [5, 7, 12, 24 , 30, 60];
        self.mMovingMedianLengths = [5, 7, 12, 24 , 30, 60];
        
        self.mCycleLengths = [5, 7, 12, 24 , 30, 60];

        self.mEnableSeasonals = True;
        
        self.mMaxAROrder = 256;
        self.mActiveTransformation = {};
        self.mActiveTransformation['None'] = True;
        self.mActiveTransformation['Difference'] = True;
        self.mActiveTransformation['RelativeDifference'] = True;
        self.mActiveTransformation['Integration'] = True;
        self.mActiveTransformation['Quantization'] = True;
        self.mActiveTransformation['BoxCox'] = True;
        self.mActiveTransformation['Logit'] = True;

    def enable_fast_mode(self):
        self.mEnableBoxCox = False;
        self.mEnableQuantization = False;        
        self.mQuantiles = [5, 10, 20]; # quintiles, deciles, and vingtiles;)
        self.mEnableMovingAverageTrends = False;
        self.mEnableMovingMedianTrends = False;
        self.mEnableTimeBasedTrends = True;
        self.mMovingAverageLengths = [5, 7, 12, 24 , 30, 60];
        self.mMovingMedianLengths = [5, 7, 12, 24 , 30, 60];
        
        self.mEnableCycles = True;
        self.mCycleLengths = [5, 7, 12, 24 , 30, 60];

        self.mEnableSeasonals = True;

        
        self.mEnableARModels = True;
        self.mMaxAROrder = 256;

        self.mActiveTransformation = {};
        self.mActiveTransformation['None'] = True;
        self.mActiveTransformation['Difference'] = True;
        self.mActiveTransformation['RelativeDifference'] = True;
        self.mActiveTransformation['Integration'] = True;
        self.mActiveTransformation['Quantization'] = False;
        self.mActiveTransformation['BoxCox'] = False;
        self.mActiveTransformation['Logit'] = False;
    


    def disable_all_transformations(self):
        self.mActiveTransformation = {};
        self.mActiveTransformation['None'] = False;
        self.mActiveTransformation['Difference'] = False;
        self.mActiveTransformation['RelativeDifference'] = False;
        self.mActiveTransformation['Integration'] = False;
        self.mActiveTransformation['Quantization'] = False;
        self.mActiveTransformation['BoxCox'] = False;
        self.mActiveTransformation['Logit'] = False;
