import pandas as pd
import numpy as np

class cSignalDecomposition_Options:
    
    def __init__(self):
        self.mParallelMode = False;
        self.mEnablePlots = False;
        self.mEstimRatio = 0.8;
        self.enable_fast_mode();
        self.mTimeDeltaComputationMethod = "AVG"; # can be "AVG", "MODE", "USER"
        self.mUserTimeDelta = None;
        self.mBusinessDaysOnly = False;
        self.mNoCoxBoxOrders = [];
        self.mCoxBoxOrders = [-2.0, -1.0 , 0.0,  1.0, 2.0];
        self.mExtensiveCoxBoxOrders = [-2, -1, -0.5, -0.33 , -0.25 , 0.0, 2, 1, 0.5, 0.33 , 0.25];
        self.mEnableTrends = True;
        self.mEnableCycles = True;
        self.mEnableARModels = True;
        self.mEnableARXModels = True;
        self.disableDebuggingOptions();

    def disableDebuggingOptions(self):
        self.mDebug = False;
        self.mDebugCycles = False;
        self.mDebugProfile = False;
        self.mDebugPerformance = False;
        
        
    def enable_slow_mode(self):
        self.mEnableCoxBox = True;
        # self.mCoxBoxOrders = self.mExtensiveCoxBoxOrders;
        self.mEnableQuantization = True;        
        self.mQuantiles = [5, 10, 20]; # quintiles, deciles, and vingtiles;)
        self.mEnableMovingAverageTrends = True;
        self.mEnableMovingMedianTrends = True;
        self.mEnableDifferentiationTransforms = True;        
        self.mEnableIntegrationTransforms = True;
        self.mEnableTimeBasedTrends = True;
        self.mMovingAverageLengths = [5, 7, 12, 24 , 30, 60];
        self.mMovingMedianLengths = [5, 7, 12, 24 , 30, 60];
        
        self.mCycleLengths = [5, 7, 12, 24 , 30, 60];
        self.mCycle_Criterion = "MAPE";
        self.mCycle_Criterion_Threshold = None;

        self.mEnableSeasonals = True;

        
        self.mMaxAROrder = 256;

    def enable_fast_mode(self):
        self.mEnableCoxBox = False;
        self.mEnableQuantization = False;        
        self.mQuantiles = [5, 10, 20]; # quintiles, deciles, and vingtiles;)
        self.mEnableMovingAverageTrends = False;
        self.mEnableMovingMedianTrends = False;
        self.mEnableDifferentiationTransforms = True;        
        self.mEnableIntegrationTransforms = True;
        self.mEnableTimeBasedTrends = True;
        self.mMovingAverageLengths = [5, 7, 12, 24 , 30, 60];
        self.mMovingMedianLengths = [5, 7, 12, 24 , 30, 60];
        
        self.mEnableCycles = True;
        self.mCycleLengths = [5, 7, 12, 24 , 30, 60];
        self.mCycle_Criterion = "MAPE";
        self.mCycle_Criterion_Threshold = None;

        self.mEnableSeasonals = True;

        
        self.mEnableARModels = True;
        self.mMaxAROrder = 256;

    
