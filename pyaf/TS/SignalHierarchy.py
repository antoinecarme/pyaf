# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
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

from multiprocessing import Pool
# from pathos.pools import ProcessPool as Pool

# for timing
import time, copy

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
            
    def to_json(self, iWithOptions = False):
        lDict = {};
        lDict['Structure'] = self.mStructure;
        lDict['Models'] = {};
        for level in sorted(self.mModels.keys()):
            for signal in sorted(self.mModels[level].keys()):
                lEngine = self.mModels[level][signal];
                lDict['Models'][signal] = lEngine.mSignalDecomposition.mBestModel.to_json(iWithOptions = False);
        if(iWithOptions):
            lDict["Options"] = self.mTimeInfo.mOptions.__dict__
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
        type1 = np.dtype(df[self.mDateColumn])
        # print(type1)
        if(type1.kind != 'M' and type1.kind != 'i' and type1.kind != 'u' and type1.kind != 'f'):
            raise tsutil.PyAF_Error("PYAF_ERROR_TIME_COLUMN_TYPE_NOT_ALLOWED '" + str(self.mDateColumn) + "' '" + str(type1) + "'");
        # level 0 is the original/physical columns
        for k in self.mStructure[0]:
            if(k not in df.columns) :
                raise tsutil.PyAF_Error("PYAF_ERROR_HIERARCHY_BASE_COLUMN_NOT_FOUND " + str(k));
            # print(type2)
            type2 = np.dtype(df[k])
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

    def train_one_model(self, arg):
        (level, signal, iAllLevelsDataset , iDateColumn , signal, H, iExogenousData, iOptions) = arg
        lEngine = autof.cForecastEngine()
        lEngine.mOptions = copy.copy(iOptions);
        lEngine.mOptions.mParallelMode = False
        lDateColumn = self.get_specific_date_column_for_signal(level, signal)
        lDateColumn = lDateColumn or iDateColumn
        lTrainDataset = iAllLevelsDataset[[lDateColumn, signal]]
        if(self.discard_nans_in_aggregate_signals()):
            lTrainDataset = lTrainDataset.dropna()
            H = self.get_horizon(level, signal)
        # print(level, signal, lTrainDataset.head())
        lEngine.train(lTrainDataset, lDateColumn , signal, H, iExogenousData = iExogenousData);
        return (level, signal, lEngine)

    def create_all_levels_models(self, iAllLevelsDataset, H, iDateColumn):
        logger = tsutil.get_pyaf_hierarchical_logger();
        args = []
        self.mModels = {};
        for level in sorted(self.mStructure.keys()):
            self.mModels[level] = {};
            for signal in sorted(self.mStructure[level].keys()):
                lExogenousData = self.get_exogenous_data(signal)
                arg = (level, signal, iAllLevelsDataset , iDateColumn , signal, H, lExogenousData, self.mOptions)
                args.append(arg)

        logger.info("TRAINING_HIERARCHICAL_MODELS_LEVEL_SIGNAL " + str([(arg[0] , arg[1]) for arg in args]));
        pool = Pool(self.mOptions.mNbCores)
        try:
            for res in pool.map(self.train_one_model, args):
                (level, signal, lEngine) = res
                # lEngine.getModelInfo();
                self.mModels[level][signal] = lEngine;
        
            pool.close()
            pool.join()
        except:
            pool.close()
            pool.join()
            raise
            
        del pool
        # print("CREATED_MODELS", self.mLevels, self.mModels)
        pass


    def fit(self):
        logger = tsutil.get_pyaf_logger();
        logger.info("START_HIERARCHICAL_TRAINING")
        start_time = time.time()
        self.create_HierarchicalStructure();
        # self.plot();
        self.create_SummingMatrix();
        lAllLevelsDataset = self.create_all_levels_dataset(self.mTrainingDataset);
        self.create_all_levels_models(lAllLevelsDataset, self.mHorizon, self.mDateColumn);
        self.computeTopDownHistoricalProportions(lAllLevelsDataset);
        lForecast_DF = self.internal_forecast(self.mTrainingDataset , self.mHorizon)
        self.computePerfOnCombinedForecasts(lForecast_DF.head(lForecast_DF.shape[0] - self.mHorizon));
        lTrainTime = time.time() - start_time;
        logger.info("END_HIERARCHICAL_TRAINING_TIME_IN_SECONDS " + str(lTrainTime))


    def getModelInfo(self):
        for level in sorted(self.mModels.keys()):
            for signal in sorted(self.mModels[level].keys()):
                lEngine = self.mModels[level][signal];
                lEngine.getModelInfo();

    def plot(self , name = None):
        logger = tsutil.get_pyaf_logger();
        logger.info("START_HIERARCHICAL_PLOTTING")
        start_time = time.time()
        lAnnotations = None;
        lHasModels = (self.mModels is not None)
        if(lHasModels):
            lAnnotations = {};
            for level in sorted(self.mStructure.keys()):
                for signal in sorted(self.mStructure[level].keys()):
                    lEngine = self.mModels[level][signal];
                    lMAPE = lEngine.mSignalDecomposition.mBestModel.mForecastPerf.mMAPE;
                    lMAPE = ('MAPE = %.4f' % lMAPE);
                    lAnnotations[signal] = [signal , lMAPE];
                    for col1 in sorted(self.mStructure[level][signal]):
                        lProp = self.mAvgHistProp[signal][col1] * 100;
                        lAnnotations[str(signal) +"_" + col1] = ('%2.2f %%' % lProp)
        tsplot.plot_hierarchy(self.mStructure, lAnnotations, name)
        lPlotTime = time.time() - start_time;
        logger.info("END_HIERARCHICAL_PLOTTING_TIME_IN_SECONDS " + str(lPlotTime))

    
    def standardPlots(self , name = None):
        for level in sorted(self.mModels.keys()):
            for signal in sorted(self.mModels[level].keys()):
                lEngine = self.mModels[level][signal];
                lEngine.standardPlots(name + "_Hierarchy_Level_Signal_" + str(level) + "_" + str(signal));

    def forecast_one_model(self, arg):
        (level, signal, lEngine, dfapp_in, H) = arg
        dfapp_out = lEngine.forecast(dfapp_in, H);
        return (level, signal, dfapp_out)

    def forecastAllModels(self, iAllLevelsDataset, H, iDateColumn):
        logger = tsutil.get_pyaf_hierarchical_logger();
        args = []
        for level in sorted(self.mModels.keys()):
            for signal in sorted(self.mModels[level].keys()):
                lEngine = self.mModels[level][signal];
                lDateColumn = self.get_specific_date_column_for_signal(level, signal)
                lDateColumn = lDateColumn or iDateColumn
                lApplyInDataset = iAllLevelsDataset[[lDateColumn, signal]]
                if(self.discard_nans_in_aggregate_signals()):
                    lApplyInDataset = lApplyInDataset.dropna()
                # print(level, signal, lApplyInDataset.head())
                dfapp_in = lApplyInDataset.copy();
                # dfapp_in.tail()
                arg = (level, signal, lEngine, dfapp_in, H)
                args.append(arg)

        logger.info("FORECASTING_HIERARCHICAL_MODELS_LEVEL_SIGNAL " + str([(arg[0] , arg[1]) for arg in args]));
        pool = Pool(self.mOptions.mNbCores)
        lForecastsByNode = {}
        try:
            for res in pool.imap(self.forecast_one_model, args):
                (level, signal, dfapp_out) = res
                lDateColumn = self.get_specific_date_column_for_signal(level, signal)
                lDateColumn = lDateColumn or iDateColumn
                # print("Forecast Columns " , dfapp_out.columns);
                lColumns = [lDateColumn , signal,
                            str(signal) + '_Forecast',
                            str(signal) + '_Forecast_Lower_Bound',
                            str(signal) + '_Forecast_Upper_Bound', ]
                lForecast_DF_i = dfapp_out[lColumns]
                lForecastsByNode[(level , signal)] = lForecast_DF_i
            pool.close()
            pool.join()
            del pool
        except:
            pool.close()
            pool.join()
            raise

        lForecast_DF = None
        for level in sorted(self.mModels.keys()):
            for signal in sorted(self.mModels[level].keys()):
                H_i = self.mHorizon # self.get_horizon(level, signal)
                lForecast_DF_i = lForecastsByNode[(level , signal)]
                lDateColumn = lForecast_DF_i.columns[0]
                if(lForecast_DF is None):
                    lForecast_DF = lForecast_DF_i
                    lForecast_DF[self.mDateColumn] = lForecast_DF_i[lDateColumn]
                else:
                    lForecast_DF = lForecast_DF.merge(lForecast_DF_i, left_on=self.mDateColumn,right_on=lDateColumn, how='left')
                    if(self.discard_nans_in_aggregate_signals()):
                        N = lForecast_DF.shape[0]
                        lForecast_DF.loc[0:N-H_i, signal] = lForecast_DF.loc[0:N-H_i, signal].fillna(0.0)
                        lForecast_DF[str(signal) + '_Forecast'] = lForecast_DF[str(signal) + '_Forecast'].fillna(0.0)                    
        
        # print(lForecast_DF.columns);
        # print(lForecast_DF.head(10));
        # print(lForecast_DF.tail(10));
        return lForecast_DF;

    def getEstimPart(self, df):
        level = 0;
        signal = list(self.mModels[level].keys())[0];
        lEngine = self.mModels[level][signal];
        lFrameFit = lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mSplit.getEstimPart(df);
        return lFrameFit;

    def getValidPart(self, df):
        level = 0;
        signal = list(self.mModels[level].keys())[0];
        lEngine = self.mModels[level][signal];
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
        lForecast_DF_BU = iForecast_DF;
        # print("STRUCTURE " , self.mStructure.keys());
        for level in sorted(self.mStructure.keys()):
            for signal in sorted(self.mStructure[level].keys()):
                new_BU_forecast = self.computeBottomUpForecast(lForecast_DF_BU, level, signal);
                lForecast_DF_BU[str(signal) + "_BU_Forecast"] = new_BU_forecast;
            
        # print(lForecast_DF_BU.head());
        # print(lForecast_DF_BU.tail());

        return lForecast_DF_BU;


    def computePerfOnCombinedForecasts(self, iForecast_DF):
        logger = tsutil.get_pyaf_hierarchical_logger();
        logger.info("FORECASTING_HIERARCHICAL_MODEL_OPTIMAL_COMBINATION_METHOD");

        self.mEstimPerfs = {}
        self.mValidPerfs = {}
        lCombinationMethods = self.mOptions.mHierarchicalCombinationMethod;
        if type(lCombinationMethods) is not list:
            lCombinationMethods = [lCombinationMethods];
        lPrefixes = [lPrefix for lPrefix in lCombinationMethods if (lPrefix != 'TD')];
        if('TD' in lCombinationMethods):
            lPrefixes = lPrefixes + ['AHP_TD', 'PHA_TD'];
        lPerfs = {};
        lFrameFit = self.getEstimPart(iForecast_DF);
        lFrameValid = self.getValidPart(iForecast_DF);
        logger.info("STRUCTURE " + str(sorted(list(self.mStructure.keys()))));
        logger.info("DATASET_COLUMNS "  + str(iForecast_DF.columns));
        for level in sorted(self.mStructure.keys()):
            logger.info("STRUCTURE_LEVEL " + str((level, sorted(list(self.mStructure[level].keys())))));
            logger.info("MODEL_LEVEL " + str((level, sorted(list(self.mModels[level].keys())))));
            for signal in sorted(self.mStructure[level].keys()):
                lEngine = self.mModels[level][signal];
                lPerfFit = lEngine.computePerf(lFrameFit[signal], lFrameFit[str(signal) + "_Forecast"], signal)
                lPerfValid = lEngine.computePerf(lFrameValid[signal], lFrameValid[str(signal) + "_Forecast"], signal)
                self.mEstimPerfs[str(signal) + "_Forecast"] = lPerfFit
                self.mValidPerfs[str(signal) + "_Forecast"] = lPerfValid
                for iPrefix in lPrefixes:
                    lPerfFit_Combined = lEngine.computePerf(lFrameFit[signal], lFrameFit[str(signal) + "_" + iPrefix + "_Forecast"],
                                                            str(signal) + "_" + iPrefix + "_Forecast")
                    lPerfValid_Combined = lEngine.computePerf(lFrameValid[signal], lFrameValid[str(signal) + "_" + iPrefix + "_Forecast"],
                                                              str(signal) + "_" + iPrefix + "_Forecast")
                    lPerfs[str(signal) + "_" + iPrefix] = (lPerfFit , lPerfValid, lPerfFit_Combined, lPerfValid_Combined);
                    self.mEstimPerfs[str(signal) + "_" + iPrefix + "_Forecast"] = lPerfFit_Combined
                    self.mValidPerfs[str(signal) + "_" + iPrefix + "_Forecast"] = lPerfValid_Combined
                                
        for (sig , perf) in sorted(lPerfs.items()):
            logger.info("REPORT_COMBINED_FORECASTS_FIT_PERF "  + str(perf[2].to_dict()))
            logger.info("REPORT_COMBINED_FORECASTS_VALID_PERF " + str(perf[3].to_dict()))
        return lPerfs;


    def computeTopDownForecasts(self, iForecast_DF , iProp , iPrefix):
        logger = tsutil.get_pyaf_hierarchical_logger();
        logger.info("FORECASTING_HIERARCHICAL_MODEL_TOP_DOWN_METHOD " + iPrefix);
        lForecast_DF_TD = iForecast_DF;
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
        lForecast_DF_MO = iForecast_DF;
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
        lForecast_DF = self.forecastAllModels(lAllLevelsDataset, iHorizon, self.mDateColumn);
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
        logger = tsutil.get_pyaf_logger();
        logger.info("START_HIERARCHICAL_FORECASTING")
        start_time = time.time()

        lForecast_DF = self.internal_forecast(iInputDS , iHorizon)

        lForecastTime = time.time() - start_time;
        logger.info("END_HIERARCHICAL_FORECAST_TIME_IN_SECONDS " + str(lForecastTime))
        return lForecast_DF;
