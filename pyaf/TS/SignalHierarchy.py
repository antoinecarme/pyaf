# Copyright (C) 2016 Antoine Carme <Antoine.Carme@outlook.com>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np

from .. import ForecastEngine as autof
from . import Options as tsopts
from . import Perf as tsperf
from . import Utils as tsutil
from . import Plots as tsplot

import copy

class cSignalHierarchy:

    def __init__(self):
        self.mHierarchy = None;
        self.mDateColumn = None;
        self.mSignal = None;
        self.mHorizon = None;
        self.mExogenousData = None;        
        self.mTrainingDataset = None;        
        self.mOptions = None;
        
        self.mLevels = None;
        self.mStructure = None;
        self.mSummingMatrix = None;

        self.mModels = None;
        
    def info(self):
        lStr2 = ""
        return lStr2;

    def get_exogenous_data(self, signal):
        if(self.mExogenousData is None):
            return None
        # A signal is a hierarchy node
        if(type(self.mExogenousData) == tuple):
            # same data for all signals
            return self.mExogenousData
        if(type(self.mExogenousData) == dict):
            # one exogenous data by signal
            return self.mExogenousData.get(signal)
        raise tsutil.PyAF_Error("BAD_EXOGENOUS_DATA_SPECIFICATION");
            
    def to_dict(self, iWithOptions = False):
        lDict = {};
        lDict['Structure'] = self.mStructure;
        lDict['Models'] = self.mModels.to_dict(iWithOptions = False)
        lDict['Models'].pop('Training_Time')
        if(iWithOptions):
            lDict["Options"] = self.mTimeInfo.mOptions.__dict__
        lDict["Training_Time"] = self.mTrainingTime
        return lDict;

    def discard_nans_in_aggregate_signals(self):
        return False
    
    def create_HierarchicalStructure(self):
        self.mLevels = self.mHierarchy['Levels'];
        self.mStructure = {};
        df = self.mHierarchy['Data'];
        lLevelCount = len(self.mLevels);
        for level in range(lLevelCount):
            self.mStructure[level] = {};
        for row in range(df.shape[0]):
            for level in range(lLevelCount):
                col = df[df.columns[level]][row];
                if(col not in self.mStructure[level].keys()):
                    self.mStructure[level][col] = set();
                if(level > 0):
                    col1 = df[df.columns[level - 1]][row];
                    self.mStructure[level][col].add(col1);
        # Stabilize the order of nodes
        for level in  sorted(self.mStructure.keys()):
            for col in sorted(self.mStructure[level].keys()):
                self.mStructure[level][col] = sorted(self.mStructure[level][col])
                    
        # print(self.mStructure);
        pass
    
    def create_SummingMatrix(self):
        lNbNodes = sum([len(self.mStructure[level]) for level in self.mStructure.keys()]);
        lBaseLevelCount = len(self.mStructure[0]);
        lIndices = {};
        self.mSummingMatrix = np.zeros((lNbNodes, lBaseLevelCount));
        for level in  sorted(self.mStructure.keys()):
            if(level > 0):
                for col in sorted(self.mStructure[level].keys()):
                    i = len(lIndices);
                    lIndices[ col ] = i;
                    for col1 in sorted(self.mStructure[level][col]):
                        ii = lIndices [ col1 ];
                        for j in range(lBaseLevelCount):
                            self.mSummingMatrix[ i ][j] = self.mSummingMatrix[ ii ][j]  + self.mSummingMatrix[ i ][j];
            else:
                for col in sorted(self.mStructure[level].keys()):
                    lNew_index = len(lIndices);
                    lIndices[ col ] = lNew_index;
                    self.mSummingMatrix[ lNew_index ] [ lNew_index ] = 1;
        # print(self.mSummingMatrix);
        self.mSummingMatrixInverse = np.linalg.pinv(self.mSummingMatrix);
        # print(self.mSummingMatrixInverse);

    def checkData(self , df):
        if(self.mHorizon != int(self.mHorizon)):
            raise tsutil.PyAF_Error("PYAF_ERROR_NON_INTEGER_HORIZON " + str(self.mHorizon));
        if(self.mHorizon < 1):
            raise tsutil.PyAF_Error("PYAF_ERROR_NEGATIVE_OR_NULL_HORIZON " + str(self.mHorizon));
        if(self.mDateColumn not in df.columns):
            raise tsutil.PyAF_Error("PYAF_ERROR_HIERARCHY_TIME_COLUMN_NOT_FOUND " + str(self.mDateColumn));
        type1 = df[self.mDateColumn].dtype
        # print(type1)
        if(type1.kind != 'M' and type1.kind != 'i' and type1.kind != 'u' and type1.kind != 'f'):
            raise tsutil.PyAF_Error("PYAF_ERROR_TIME_COLUMN_TYPE_NOT_ALLOWED '" + str(self.mDateColumn) + "' '" + str(type1) + "'");
        # level 0 is the original/physical columns
        for k in self.mStructure[0]:
            if(k not in df.columns) :
                raise tsutil.PyAF_Error("PYAF_ERROR_HIERARCHY_BASE_COLUMN_NOT_FOUND " + str(k));
            # print(type2)
            type2 = df[k].dtype
            if(type2.kind != 'i' and type2.kind != 'u' and type2.kind != 'f'):
                raise tsutil.PyAF_Error("PYAF_ERROR_HIERARCHY_BASE_SIGNAL_COLUMN_TYPE_NOT_ALLOWED '" + str(k) + "' '" + str(type2) + "'");


    def create_all_levels_dataset(self, df):
        self.checkData(df);
        lAllLevelsDataset = df.copy();
        lMapped = True;
        # level 0 is the original/physical columns
        for k in self.mStructure[0]:
            if(k not in df.columns) :
                lMapped = False;
        if(not lMapped):
            i = 0;
            for k in self.mStructure[0]:
                print("MAPPING_ORIGINAL_COLUMN" , df.columns[i + 1], "=>" , k)
                lAllLevelsDataset[k] = df[df.columns[i + 1]];
                i = i + 1;
                
        for level in  sorted(self.mStructure.keys()):
            if(level > 0):
                for col in sorted(self.mStructure[level].keys()):
                    new_col = None;
                    for col1 in sorted(self.mStructure[level][col]):
                        if(new_col is None):
                            new_col = lAllLevelsDataset[col1];
                        else:
                            new_col = new_col + lAllLevelsDataset[col1];
                    lAllLevelsDataset[col] = new_col;
        return lAllLevelsDataset;


    def addVars(self, df):
        lAllLevelsDataset = self.create_all_levels_dataset(df);
        return lAllLevelsDataset;

    def transformDataset(self, df):
        df = self.addVars(df);
        return df;

    def get_specific_date_column_for_signal(self, level, signal):
        # only for temporal hierarchies
        return None

    def get_horizon(self, level, signal):
        # only for temporal hierarchies
        return self.mHorizon

    def create_all_levels_models_with_one_engine(self, iAllLevelsDataset, H, iDateColumn):
        logger = tsutil.get_pyaf_hierarchical_logger();
        lSignals = []
        lDateColumns = {}
        lExogenousData = {}
        lHorizons = {}
        lDiscardNulls = {}
        for level in sorted(self.mStructure.keys()):
            for signal in sorted(self.mStructure[level].keys()):
                lSignals = lSignals + [signal]
                lExogenousData[signal] = self.get_exogenous_data(signal)
                lDateColumn = self.get_specific_date_column_for_signal(level, signal)
                lDateColumns[signal] = lDateColumn or iDateColumn
                lHorizons[signal] = self.get_horizon(level, signal)

        lEngine = autof.cForecastEngine()
        lEngine.mOptions = copy.copy(self.mOptions);
        if(self.discard_nans_in_aggregate_signals()):
            lEngine.mOptions.mMissingDataOptions.mTimeMissingDataImputation = "DiscardRow"
            lEngine.mOptions.mMissingDataOptions.mSignalMissingDataImputation = "DiscardRow"
            # Sampling is not compatible with Temporal Hierarchies (#163)
            lEngine.mOptions.mActivateSampling = False
        assert(iAllLevelsDataset.shape[0] > 0)
        lEngine.train(iAllLevelsDataset, lDateColumns , lSignals, lHorizons, iExogenousData = lExogenousData);
        self.mModels = lEngine
        # print("CREATED_MODELS", self.mLevels, self.mModels)


    def fit(self):
        lTimer = tsutil.cTimer(("HIERARCHICAL_TRAINING"))
        self.create_HierarchicalStructure();
        # self.plot();
        self.create_SummingMatrix();
        lAllLevelsDataset = self.create_all_levels_dataset(self.mTrainingDataset);
        self.create_all_levels_models_with_one_engine(lAllLevelsDataset, self.mHorizon, self.mDateColumn);
        self.computeTopDownHistoricalProportions(lAllLevelsDataset);
        lForecast_DF = self.internal_forecast(self.mTrainingDataset , self.mHorizon)
        self.computePerfOnCombinedForecasts(lForecast_DF.head(lForecast_DF.shape[0] - self.mHorizon));
        self.mTrainingTime = lTimer.get_elapsed_time()


    def getModelInfo(self):
        lEngine = self.mModels
        lEngine.getModelInfo();


    def get_plot_annotations(self):
        lAnnotations = None;
        lHasModels = (self.mModels is not None)
        if(lHasModels):
            lPrefixes = self.get_reconciled_forecast_prefixes()
            lAnnotations = {};
            for level in sorted(self.mStructure.keys()):
                for signal in sorted(self.mStructure[level].keys()):
                    lEngine = self.mModels
                    lMAPE = 'MAPE = %.4f' % self.mValidPerfs[str(signal) + "_Forecast"].mMAPE
                    lReconciledMAPEs = [ ]
                    for lPrefix in lPrefixes:
                        lMAPE_Rec = self.mValidPerfs[str(signal) + "_" + lPrefix + "_Forecast"].mMAPE
                        lReconciledMAPEs.append('MAPE_' + lPrefix + ' = %.4f' % lMAPE_Rec);
                    lAnnotations[signal] = [signal , lMAPE ] + lReconciledMAPEs;
                    for col1 in sorted(self.mStructure[level][signal]):
                        lProp = self.mAvgHistProp[signal][col1] * 100;
                        lAnnotations[str(signal) +"_" + col1] = ('%2.2f %%' % lProp)
        return lAnnotations

    def plot(self , name = None):
        lTimer = tsutil.cTimer(("HIERARCHICAL_PLOTTING"))
        lAnnotations = self.get_plot_annotations()
        tsplot.plot_hierarchy(self.mStructure, lAnnotations, name)

    def plot_as_png_base64(self , name = None):
        lTimer = tsutil.cTimer(("HIERARCHICAL_PLOTTING_AS_PNG"))
        lAnnotations = self.get_plot_annotations()
        lBase64 = tsplot.plot_hierarchy_as_png_base64(self.mStructure, lAnnotations, name)
        return lBase64
    
    def standardPlots(self , name = None):
        lEngine = self.mModels
        lEngine.standardPlots(name + "_Hierarchy_Level_Signal_");
        self.plot(name + "_Hierarchical_Structure.png")

    def getPlotsAsDict(self):
        lDict = {}
        lDict["Models"] = self.mModels.getPlotsAsDict()
        lDict["Hierarchical_Structure"] = self.plot_as_png_base64()
        return lDict
    

    def forecastAllModels_with_one_engine(self, iAllLevelsDataset, H, iDateColumn):
        logger = tsutil.get_pyaf_hierarchical_logger();
        lEngine = self.mModels
        lForecast_DF = lEngine.forecast(iAllLevelsDataset, H);
        lDateColumns = []
        lSigColumns = []
        for signal in lEngine.mSignalDecomposition.mSignals:
            lDateColumn = lEngine.mSignalDecomposition.mDateColumns[signal]
            lDateColumns = lDateColumns + [lDateColumn]
            lSigColumns = lSigColumns + [signal ,
                                         str(signal) + '_Forecast',
                                         str(signal) + '_Forecast_Lower_Bound',
                                         str(signal) + '_Forecast_Upper_Bound']
        lColumns = list(set(lDateColumns)) + lSigColumns
        if(self.discard_nans_in_aggregate_signals()):
            H = self.mHorizon
            N = lForecast_DF.shape[0]
            lForecast_DF.loc[0:N-H, signal] = lForecast_DF.loc[0:N-H, signal].fillna(0.0)
            lForecast_DF[str(signal) + '_Forecast'] = lForecast_DF[str(signal) + '_Forecast'].fillna(0.0)     
        return lForecast_DF[lColumns]
    
    def getEstimPart(self, df):
        lEngine = self.mModels
        lFrameFit = lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mSplit.getEstimPart(df);
        return lFrameFit;

    def getValidPart(self, df):
        lEngine = self.mModels
        lFrameFit = lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mSplit.getValidPart(df);
        return lFrameFit;


    def computeTopDownHistoricalProportions(self, iAllLevelsDataset):
        logger = tsutil.get_pyaf_hierarchical_logger();
        logger.info("TRAINING_HIERARCHICAL_MODEL_COMPUTE_TOP_DOWN_HISTORICAL_PROPORTIONS");
        self.mAvgHistProp = {};
        self.mPropHistAvg = {};
        # Compute these proportions only on Estimation.
        lEstim = self.getEstimPart(iAllLevelsDataset);
        for level in  sorted(self.mStructure.keys()):
            if(level > 0):
                for col in sorted(self.mStructure[level].keys()):
                    self.mAvgHistProp[col] = {};
                    self.mPropHistAvg[col] = {};
                    for col1 in sorted(self.mStructure[level][col]):
                        self.mAvgHistProp[col][col1] = (lEstim[col1] / lEstim[col]).mean();
                        self.mPropHistAvg[col][col1] = lEstim[col1].mean() / lEstim[col].mean();
        # print("AvgHitProp\n", self.mAvgHistProp);
        # print("PropHistAvg\n", self.mPropHistAvg);
        pass
        
    def computeTopDownForecastedProportions(self, iForecast_DF):
        self.mForecastedProp = {};
        for level in  sorted(self.mStructure.keys()):
            if(level > 0):
                for col in sorted(self.mStructure[level].keys()):
                    self.mForecastedProp[col] = {};
                    for col1 in sorted(self.mStructure[level][col]):
                        self.mForecastedProp[col][col1] = (iForecast_DF[col1] / iForecast_DF[col]).mean();
        # print("ForecastedProp\n", self.mForecastedProp);
        pass

    def computeBottomUpForecast(self, iForecast_DF, level, signal, iPrefix = "BU"):
        new_BU_forecast = None;
        for col1 in sorted(self.mStructure[level][signal]):
            if(new_BU_forecast is None):
                new_BU_forecast = iForecast_DF[col1 + "_Forecast"];
            else:
                new_BU_forecast = new_BU_forecast + iForecast_DF[col1 + "_" + iPrefix + "_Forecast"];
        if(new_BU_forecast is None):
            new_BU_forecast = iForecast_DF[str(signal) + "_Forecast"];
        return new_BU_forecast;

    def computeBottomUpForecasts(self, iForecast_DF):
        logger = tsutil.get_pyaf_hierarchical_logger();
        logger.info("FORECASTING_HIERARCHICAL_MODEL_BOTTOM_UP_METHOD " + "BU");
        lForecast_DF_BU = iForecast_DF.copy()
        # print("STRUCTURE " , self.mStructure.keys());
        for level in sorted(self.mStructure.keys()):
            for signal in sorted(self.mStructure[level].keys()):
                new_BU_forecast = self.computeBottomUpForecast(lForecast_DF_BU, level, signal);
                lForecast_DF_BU[str(signal) + "_BU_Forecast"] = new_BU_forecast;
            
        # print(lForecast_DF_BU.head());
        # print(lForecast_DF_BU.tail());

        return lForecast_DF_BU;

    def get_clean_signal_and_forecasts(self, iForecast_DF, signal, iPrefixes):
        lEngine = self.mModels
        lForecasts = [str(signal) + "_Forecast"]
        lForecasts = lForecasts + [str(signal) + "_" + lPrefix + "_Forecast" for lPrefix in iPrefixes]
        lColumns = [lEngine.mSignalDecomposition.mDateColumns[signal] , signal ] + lForecasts
        lForecast_DF = iForecast_DF[lColumns]
        return lForecast_DF

    def get_reconciled_forecast_prefixes(self):
        lCombinationMethods = self.mOptions.mHierarchicalCombinationMethod;
        if type(lCombinationMethods) is not list:
            lCombinationMethods = [lCombinationMethods];
        lPrefixes = [lPrefix for lPrefix in lCombinationMethods if (lPrefix != 'TD')];
        if('TD' in lCombinationMethods):
            lPrefixes = lPrefixes + ['AHP_TD', 'PHA_TD'];
        return lPrefixes

    def computePerfOnCombinedForecasts(self, iForecast_DF):
        logger = tsutil.get_pyaf_hierarchical_logger();
        logger.info("FORECASTING_HIERARCHICAL_MODEL_OPTIMAL_COMBINATION_METHOD");
        lEngine = self.mModels
        lPrefixes = self.get_reconciled_forecast_prefixes()
        
        self.mEstimPerfs = {}
        self.mValidPerfs = {}
        lPerfs = {};
        logger.info("STRUCTURE " + str(sorted(list(self.mStructure.keys()))));
        logger.info("DATASET_COLUMNS "  + str(iForecast_DF.columns));
        for level in sorted(self.mStructure.keys()):
            logger.info("STRUCTURE_LEVEL " + str((level, sorted(list(self.mStructure[level].keys())))));
            for signal in sorted(self.mStructure[level].keys()):
                lForecast_DF = self.get_clean_signal_and_forecasts(iForecast_DF, signal, lPrefixes)                
                lFrameFit = self.getEstimPart(lForecast_DF);
                lFrameValid = self.getValidPart(lForecast_DF);
                lColumns = [signal , str(signal) + "_Forecast"] + [str(signal) + "_" + lPrefix + "_Forecast" for lPrefix in lPrefixes]
                lFrameFit = lFrameFit[lColumns]
                lFrameValid = lFrameValid[lColumns]
                if(self.discard_nans_in_aggregate_signals()):
                    lFrameFit = lFrameFit.dropna()
                    lFrameValid = lFrameValid.dropna()
                lPerfFit = lEngine.computePerf(lFrameFit[signal], lFrameFit[str(signal) + "_Forecast"], signal)
                lPerfValid = lEngine.computePerf(lFrameValid[signal], lFrameValid[str(signal) + "_Forecast"], signal)
                self.mEstimPerfs[str(signal) + "_Forecast"] = lPerfFit
                self.mValidPerfs[str(signal) + "_Forecast"] = lPerfValid
                for iPrefix in lPrefixes:
                    lName = str(signal) + "_" + iPrefix + "_Forecast"
                    lPerfFit_Combined = lEngine.computePerf(lFrameFit[signal], lFrameFit[lName], lName)
                    lPerfValid_Combined = lEngine.computePerf(lFrameValid[signal], lFrameValid[lName], lName)
                    lPerfs[str(signal) + "_" + iPrefix] = (lPerfFit , lPerfValid, lPerfFit_Combined, lPerfValid_Combined);
                    self.mEstimPerfs[lName] = lPerfFit_Combined
                    self.mValidPerfs[lName] = lPerfValid_Combined
                                
        for (sig , perf) in sorted(lPerfs.items()):
            logger.info("REPORT_COMBINED_FORECASTS_FIT_PERF "  + str(perf[2].to_dict()))
            logger.info("REPORT_COMBINED_FORECASTS_VALID_PERF " + str(perf[3].to_dict()))
        return lPerfs;


    def computeTopDownForecasts(self, iForecast_DF , iProp , iPrefix):
        logger = tsutil.get_pyaf_hierarchical_logger();
        logger.info("FORECASTING_HIERARCHICAL_MODEL_TOP_DOWN_METHOD " + iPrefix);
        lForecast_DF_TD = iForecast_DF.copy()
        lLevelsReversed = sorted(self.mStructure.keys(), reverse=True);
        # print("TOPDOWN_STRUCTURE", self.mStructure)
        # print("TOPDOWN_LEVELS", lLevelsReversed)
        # highest levels (fully aggregated)
        lHighestLevel = lLevelsReversed[0];
        for signal in sorted(self.mStructure[lHighestLevel].keys()):
            lForecast_DF_TD[str(signal) +"_" + iPrefix + "_Forecast"] =  iForecast_DF[str(signal) + "_Forecast"];
        for level in lLevelsReversed:
            for signal in sorted(self.mStructure[level].keys()):
                for col in sorted(self.mStructure[level][signal]):
                    new_TD_forecast = lForecast_DF_TD[str(signal) + "_" + iPrefix + "_Forecast"] * iProp[signal][col];
                    lForecast_DF_TD[str(col) +"_" + iPrefix + "_Forecast"] = new_TD_forecast;
        
        # print(lForecast_DF_TD.head());
        # print(lForecast_DF_TD.tail());

        return lForecast_DF_TD;

    def computeMiddleOutForecasts(self, iForecast_DF , iProp, iPrefix):
        logger = tsutil.get_pyaf_hierarchical_logger();
        logger.info("FORECASTING_HIERARCHICAL_MODEL_MIDDLE_OUT_METHOD " + iPrefix);
        lLevels = self.mStructure.keys();
        lMidLevel = len(lLevels) // 2;
        lForecast_DF_MO = iForecast_DF.copy()
        # lower levels .... top-down starting from the middle.
        levels_below = sorted([level for level in self.mStructure.keys()  if (level <= lMidLevel) ],
                              reverse=True);
        # print("MIDDLE_OUT_STRUCTURE", self.mStructure)
        # print("MIDDLE_OUT_LEVELS", levels_below)
        # mid-lewvel : do nothing ????
        for signal in sorted(self.mStructure[lMidLevel].keys()):
            lForecast_DF_MO[str(signal) +"_" + iPrefix + "_Forecast"] = iForecast_DF[str(signal) + "_Forecast"];
        # 
        for level in levels_below:
            for signal in sorted(self.mStructure[level].keys()):
                for col in sorted(self.mStructure[level][signal]):
                    new_MO_forecast = lForecast_DF_MO[str(signal) + "_" + iPrefix + "_Forecast"] * iProp[signal][col];
                    lForecast_DF_MO[str(col) +"_" + iPrefix + "_Forecast"] = new_MO_forecast;
        # higher levels .... bottom-up starting from the middle
        for level in range(lMidLevel + 1 , len(self.mStructure.keys())):
            for signal in sorted(self.mStructure[level].keys()):
                new_MO_forecast = self.computeBottomUpForecast(lForecast_DF_MO, level, signal, iPrefix);
                lForecast_DF_MO[str(signal) + "_" + iPrefix + "_Forecast"] = new_MO_forecast;

        # print(lForecast_DF_MO.head());
        # print(lForecast_DF_MO.tail());

        return lForecast_DF_MO;


    def computeOptimalCombination(self, iForecast_DF):
        logger = tsutil.get_pyaf_hierarchical_logger();
        logger.info("FORECASTING_HIERARCHICAL_MODEL_OPTIMAL_COMBINATION_METHOD " + "OC");
        lBaseNames = [];
        for level in  sorted(self.mStructure.keys()):
            for col in sorted(self.mStructure[level].keys()):
                lBaseNames.append(col);
        lBaseForecastNames = [str(col) + "_Forecast" for col in lBaseNames]
        lBaseForecasts = iForecast_DF[lBaseForecastNames];
        # TODO : use linalg.solve here
        S = self.mSummingMatrix;
        # print(S.shape);
        lInv = np.linalg.inv(S.T.dot(S))
        lOptimalForecasts = S.dot(lInv).dot(S.T).dot(lBaseForecasts.values.T)
        # print(lBaseForecasts.shape);
        # print(lOptimalForecasts.shape);
        lOptimalNames = [(str(col) + "_OC_Forecast") for col in lBaseNames];
        df = pd.DataFrame(lOptimalForecasts.T);
        df.columns = lOptimalNames;
        lForecast_DF_OC = pd.concat([iForecast_DF , df] , axis = 1);
        
        # print(lForecast_DF_OC.head());
        # print(lForecast_DF_OC.tail());
        return lForecast_DF_OC;

    def internal_forecast(self , iInputDS, iHorizon):

        lAllLevelsDataset = self.create_all_levels_dataset(iInputDS);
        lForecast_DF = self.forecastAllModels_with_one_engine(lAllLevelsDataset, iHorizon, self.mDateColumn);
        lCombinationMethods = self.mOptions.mHierarchicalCombinationMethod;
        if type(lCombinationMethods) is not list:
            lCombinationMethods = [lCombinationMethods];
        logger = tsutil.get_pyaf_hierarchical_logger();
        logger.info("FORECASTING_HIERARCHICAL_MODEL_COMBINATION_METHODS " + str(lCombinationMethods));

        for lMethod in lCombinationMethods:
            if(lMethod == "BU"):            
                lForecast_DF_BU = self.computeBottomUpForecasts(lForecast_DF);
                lForecast_DF = lForecast_DF_BU;
        
            if(lMethod == "TD"):            
                lForecast_DF_TD_AHP = self.computeTopDownForecasts(lForecast_DF, self.mAvgHistProp, "AHP_TD") 
                lForecast_DF = lForecast_DF_TD_AHP;
                
                lForecast_DF_TD_PHA = self.computeTopDownForecasts(lForecast_DF, self.mPropHistAvg, "PHA_TD")
                lForecast_DF = lForecast_DF_TD_PHA;
        
            if(lMethod == "MO"):            
                lForecast_DF_MO = self.computeMiddleOutForecasts(lForecast_DF,
                                                                 self.mPropHistAvg,
                                                                 "MO")
                lForecast_DF = lForecast_DF_MO;

            if(lMethod == "OC"):            
                lForecast_DF_OC = self.computeOptimalCombination(lForecast_DF);
                lForecast_DF = lForecast_DF_OC;
        return lForecast_DF

    def forecast(self , iInputDS, iHorizon):
        lTimer = tsutil.cTimer(("HIERARCHICAL_FORECAST"))

        lForecast_DF = self.internal_forecast(iInputDS , iHorizon)

        return lForecast_DF;
