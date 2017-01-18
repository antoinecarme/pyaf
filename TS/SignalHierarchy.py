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


class cSignalHierarchy:

    def __init__(self):
        self.mHierarchy = None;
        self.mHierarchy = None;
        self.mDateColumn = None;
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


    def to_json(self):
        lDict = {};
        lDict['Structure'] = self.mStructure;
        lDict['Models'] = {};
        for level in sorted(self.mModels.keys()):
            for signal in sorted(self.mModels[level].keys()):
                lEngine = self.mModels[level][signal];
                lDict['Models'][signal] = lEngine.mSignalDecomposition.mBestModel.to_json();
        return lDict;

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

    def create_all_levels_dataset(self, df):
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


    def create_all_levels_models(self, iAllLevelsDataset, H, iDateColumn):
        logger = tsutil.get_pyaf_hierarchical_logger();
        self.mModels = {};
        for level in sorted(self.mStructure.keys()):
            self.mModels[level] = {};
            for signal in sorted(self.mStructure[level].keys()):
                logger.info("TRAINING_MODEL_LEVEL_SIGNAL " + str(level) + " " + signal);
                lEngine = autof.cForecastEngine()
                lEngine.mOptions = self.mOptions;
                lEngine.train(iAllLevelsDataset , iDateColumn , signal, H);
                lEngine.getModelInfo();
                self.mModels[level][signal] = lEngine;
        # print("CREATED_MODELS", self.mLevels, self.mModels)
        pass


    def fit(self):
        self.create_HierarchicalStructure();
        # self.plot();
        self.create_SummingMatrix();
        lAllLevelsDataset = self.create_all_levels_dataset(self.mTrainingDataset);
        self.create_all_levels_models(lAllLevelsDataset, self.mHorizon, self.mDateColumn);
        self.computeTopDownHistoricalProportions(lAllLevelsDataset);


    def getModelInfo(self):
        for level in sorted(self.mModels.keys()):
            for signal in sorted(self.mModels[level].keys()):
                lEngine = self.mModels[level][signal];
                lEngine.getModelInfo();

    def plot(self , name = None):
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
                        lAnnotations[signal +"_" + col1] = ('%2.2f %%' % lProp)
        tsplot.plot_hierarchy(self.mStructure, lAnnotations, name)
    
    def standrdPlots(self , name = None):
        for level in sorted(self.mModels.keys()):
            for signal in sorted(self.mModels[level].keys()):
                lEngine = self.mModels[level][signal];
                lEngine.standrdPlots(name + "_Hierarchy_Level_Signal_" + str(level) + "_" + signal);


    def forecastAllModels(self, iAllLevelsDataset, H, iDateColumn):
        lForecast_DF = pd.DataFrame();
        lForecast_DF[iDateColumn] = iAllLevelsDataset[iDateColumn]
        for level in sorted(self.mModels.keys()):
            for signal in sorted(self.mModels[level].keys()):
                lEngine = self.mModels[level][signal];
                dfapp_in = iAllLevelsDataset[[iDateColumn , signal]].copy();
                # dfapp_in.tail()
                dfapp_out = lEngine.forecast(dfapp_in, H);
                # print("Forecast Columns " , dfapp_out.columns);
                lForecast_DF[signal] = dfapp_out[signal]
                lForecast_DF[signal + '_Forecast'] = dfapp_out[signal + '_Forecast']
        # print(lForecast_DF.columns);
        # print(lForecast_DF.head());
        # print(lForecast_DF.tail());
        return lForecast_DF;

    def getEstimPart(self, df):
        level = 0;
        signal = list(self.mModels[level].keys())[0];
        lEngine = self.mModels[level][signal];
        lFrameFit = lEngine.mSignalDecomposition.mBestModel.mTimeInfo.getEstimPart(df);
        return lFrameFit;


    def computeTopDownHistoricalProportions(self, iAllLevelsDataset):
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
            new_BU_forecast = iForecast_DF[signal + "_Forecast"];
        return new_BU_forecast;

    def computeBottomUpForecasts(self, iForecast_DF):
        lForecast_DF_BU = iForecast_DF;
        # print("STRUCTURE " , self.mStructure.keys());
        for level in sorted(self.mStructure.keys()):
            for signal in sorted(self.mStructure[level].keys()):
                new_BU_forecast = self.computeBottomUpForecast(lForecast_DF_BU, level, signal);
                lForecast_DF_BU[signal + "_BU_Forecast"] = new_BU_forecast;
            
        # print(lForecast_DF_BU.head());
        # print(lForecast_DF_BU.tail());

        return lForecast_DF_BU;


    def reportOnCombinedForecasts(self, iForecast_DF, iPrefixes):
        lPrefixes = [lPrefix for lPrefix in iPrefixes if (lPrefix != 'TD')];
        if('TD' in iPrefixes):
            lPrefixes = lPrefixes + ['AHP_TD', 'PHA_TD'];
        lPerfs = {};
        print("STRUCTURE " , sorted(list(self.mStructure.keys())));
        print("DATASET_COLUMNS" , iForecast_DF.columns);
        for level in sorted(self.mStructure.keys()):
            print("STRUCTURE_LEVEL " , level, sorted(list(self.mStructure[level].keys())));
            print("MODEL_LEVEL " , level, sorted(list(self.mModels[level].keys())));
            for signal in sorted(self.mStructure[level].keys()):
                lEngine = self.mModels[level][signal];
                lPerf = lEngine.computePerf(iForecast_DF[signal], iForecast_DF[signal + "_Forecast"], signal)
                for iPrefix in lPrefixes:
                    lPerf_Combined = lEngine.computePerf(iForecast_DF[signal], iForecast_DF[signal + "_" + iPrefix + "_Forecast"],  signal)
                    lPerfs[signal + "_" + iPrefix] = (lPerf , lPerf_Combined);
            
        for (sig , perf) in sorted(lPerfs.items()):
            print("PERF_REPORT_COMBINED_FORECASTS" , sig , perf[0].mL2,  perf[0].mMAPE, perf[1].mL2,  perf[1].mMAPE)
        return lPerfs;


    def computeTopDownForecasts(self, iForecast_DF , iProp , iPrefix):
        lForecast_DF_TD = iForecast_DF;
        lLevelsReversed = sorted(self.mStructure.keys(), reverse=True);
        # print("TOPDOWN_STRUCTURE", self.mStructure)
        # print("TOPDOWN_LEVELS", lLevelsReversed)
        # highest levels (fully aggregated)
        lHighestLevel = lLevelsReversed[0];
        for signal in sorted(self.mStructure[lHighestLevel].keys()):
            lForecast_DF_TD[signal +"_" + iPrefix + "_Forecast"] =  iForecast_DF[signal + "_Forecast"];
        for level in lLevelsReversed:
            for signal in sorted(self.mStructure[level].keys()):
                for col in sorted(self.mStructure[level][signal]):
                    new_TD_forecast = lForecast_DF_TD[signal + "_" + iPrefix + "_Forecast"] * iProp[signal][col];
                    lForecast_DF_TD[col +"_" + iPrefix + "_Forecast"] = new_TD_forecast;
        
        # print(lForecast_DF_TD.head());
        # print(lForecast_DF_TD.tail());

        return lForecast_DF_TD;

    def computeMiddleOutForecasts(self, iForecast_DF , iProp, iPrefix):
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
            lForecast_DF_MO[signal +"_" + iPrefix + "_Forecast"] = iForecast_DF[signal + "_Forecast"];
        # 
        for level in levels_below:
            for signal in sorted(self.mStructure[level].keys()):
                for col in sorted(self.mStructure[level][signal]):
                    new_MO_forecast = lForecast_DF_MO[signal + "_" + iPrefix + "_Forecast"] * iProp[signal][col];
                    lForecast_DF_MO[col +"_" + iPrefix + "_Forecast"] = new_MO_forecast;
        # higher levels .... bottom-up starting from the middle
        for level in range(lMidLevel + 1 , len(self.mStructure.keys())):
            for signal in sorted(self.mStructure[level].keys()):
                new_MO_forecast = self.computeBottomUpForecast(lForecast_DF_MO, level, signal, iPrefix);
                lForecast_DF_MO[signal +"_" + iPrefix + "_Forecast"] = new_MO_forecast;

        # print(lForecast_DF_MO.head());
        # print(lForecast_DF_MO.tail());

        return lForecast_DF_MO;


    def computeOptimalCombination(self, iForecast_DF):
        lBaseNames = [];
        for level in  sorted(self.mStructure.keys()):
            for col in sorted(self.mStructure[level].keys()):
                lBaseNames.append(col);
        lBaseForecastNames = [col + "_Forecast" for col in lBaseNames]
        lBaseForecasts = iForecast_DF[lBaseForecastNames];
        # TODO : use linalg.solve here
        S = self.mSummingMatrix;
        # print(S.shape);
        lInv = np.linalg.inv(S.T.dot(S))
        lOptimalForecasts = S.dot(lInv).dot(S.T).dot(lBaseForecasts.values.T)
        # print(lBaseForecasts.shape);
        # print(lOptimalForecasts.shape);
        lOptimalNames = [(col + "_OC_Forecast") for col in lBaseNames];
        df = pd.DataFrame(lOptimalForecasts.T);
        df.columns = lOptimalNames;
        lForecast_DF_OC = pd.concat([iForecast_DF , df] , axis = 1);
        
        # print(lForecast_DF_OC.head());
        # print(lForecast_DF_OC.tail());
        return lForecast_DF_OC;

    def forecast(self , iInputDS, iHorizon):
        lAllLevelsDataset = self.create_all_levels_dataset(iInputDS);
        lForecast_DF = self.forecastAllModels(lAllLevelsDataset, iHorizon, self.mDateColumn);
        lCombinationMethods = self.mOptions.mHierarchicalCombinationMethod;
        if type(lCombinationMethods) is not list:
            lCombinationMethods = [lCombinationMethods];

        for lMethod in lCombinationMethods:
            if(lMethod == "BU"):            
                lForecast_DF_BU = self.computeBottomUpForecasts(lForecast_DF);
                # self.reportOnCombinedForecasts(lForecast_DF_BU , "BU");
                # lForecast_DF_BU.to_csv("outputs/aggregated_forecasts_bu.csv");
                lForecast_DF = lForecast_DF_BU;
        
            if(lMethod == "TD"):            
                lForecast_DF_TD_AHP = self.computeTopDownForecasts(lForecast_DF, self.mAvgHistProp, "AHP_TD") 
                # self.reportOnCombinedForecasts(lForecast_DF_TD_AHP , "AHP_TD");
                lForecast_DF = lForecast_DF_TD_AHP;
                
                lForecast_DF_TD_PHA = self.computeTopDownForecasts(lForecast_DF, self.mPropHistAvg, "PHA_TD")
                # self.reportOnCombinedForecasts(lForecast_DF_TD_PHA , "PHA_TD");
                lForecast_DF = lForecast_DF_TD_PHA;
                # lForecast_DF_TD_PHA.to_csv("outputs/aggregated_forecasts_td_pha.csv");
                # lForecast_DF_TD_PHA.to_csv("outputs/aggregated_forecasts_td_ahp.csv");
        
            if(lMethod == "MO"):            
                lForecast_DF_MO = self.computeMiddleOutForecasts(lForecast_DF,
                                                                 self.mPropHistAvg,
                                                                 "MO")
                # self.reportOnCombinedForecasts(lForecast_DF_MO , "MO"); 
                # lForecast_DF_MO.to_csv("outputs/aggregated_forecasts_mo.csv");
                lForecast_DF = lForecast_DF_MO;

            if(lMethod == "OC"):            
                lForecast_DF_OC = self.computeOptimalCombination(lForecast_DF);
                # self.reportOnCombinedForecasts(lForecast_DF_OC , "OC");                
                # lForecast_DF_OC.to_csv("outputs/aggregated_forecasts_oc.csv");
                lForecast_DF = lForecast_DF_OC;

        self.reportOnCombinedForecasts(lForecast_DF , lCombinationMethods);
                
        return lForecast_DF;
