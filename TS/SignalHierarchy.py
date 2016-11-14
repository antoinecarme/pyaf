# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np


import pyaf.ForecastEngine as autof
from . import Options as tsopts
from . import Perf as tsperf
from . import Utils as tsutil

from CodeGen import TS_CodeGen_Objects as tscodegen


class cSignalHierarchy:

    def __init__(self):
        self.mHierarchy = None;
        self.mDateVariable = None;
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
        dict1 = {};
        return dict1;

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
        print(self.mStructure);
        pass
    
    def create_SummingMatrix(self):
        lLevelCount = len(self.mLevels);
        lNbNodes = sum([len(self.mStructure[level]) for level in self.mStructure.keys()]);
        lBaseLevelCount = len(self.mStructure[0]);
        lIndices = {};
        self.mSummingMatrix = np.zeros((lNbNodes, lBaseLevelCount));
        for level in  self.mStructure.keys():
            if(level > 0):
                for col in self.mStructure[level].keys():
                    i = len(lIndices);
                    lIndices[ col ] = i;
                    for col1 in self.mStructure[level][col]:
                        ii = lIndices [ col1 ];
                        for j in range(lBaseLevelCount):
                            self.mSummingMatrix[ i ][j] = self.mSummingMatrix[ ii ][j]  + self.mSummingMatrix[ i ][j];
            else:
                for col in self.mStructure[level].keys():
                    lNew_index = len(lIndices);
                    lIndices[ col ] = lNew_index;
                    self.mSummingMatrix[ lNew_index ] [ lNew_index ] = 1;
        print(self.mSummingMatrix);
        self.mSummingMatrixInverse = np.linalg.pinv(self.mSummingMatrix);
        print(self.mSummingMatrixInverse);

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
                
        lLevelCount = len(self.mLevels);
        for level in  self.mStructure.keys():
            if(level > 0):
                for col in self.mStructure[level].keys():
                    new_col = None;
                    for col1 in self.mStructure[level][col]:
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
        self.mModels = {};
        lLevelCount = len(self.mLevels);
        for level in self.mStructure.keys():
            self.mModels[level] = {};
            for signal in self.mStructure[level].keys():
                lEngine = autof.cForecastEngine()
                lEngine.mOptions = self.mOptions;
                lEngine.train(iAllLevelsDataset , iDateColumn , signal, H);
                lEngine.getModelInfo();
                self.mModels[level][signal] = lEngine;
        print("CREATED_MODELS", self.mLevels, self.mModels)
        pass


    def fit(self):
        self.create_HierarchicalStructure();
        self.create_SummingMatrix();
        lAllLevelsDataset = self.create_all_levels_dataset(self.mTrainingDataset);
        self.create_all_levels_models(lAllLevelsDataset, self.mHorizon, self.mDateColumn);


    def getModelInfo(self):
        for level in self.mModels.keys():
            for signal in self.mModels[level].keys():
                lEngine = self.mModels[level][signal];
                lEngine.getModelInfo();

    
    def standrdPlots(self , name = None):
        for level in self.mModels.keys():
            for signal in self.mModels[level].keys():
                lEngine = self.mModels[level][signal];
                lEngine.standrdPlots(name + "_Hierarchy_Level_Signal_" + str(level) + "_" + signal);


    def forecastAllModels(self, iAllLevelsDataset, H, iDateColumn):
        lForecast_DF = pd.DataFrame();
        lForecast_DF[iDateColumn] = iAllLevelsDataset[iDateColumn]
        for level in self.mModels.keys():
            for signal in self.mModels[level].keys():
                lEngine = self.mModels[level][signal];
                dfapp_in = iAllLevelsDataset.copy();
                # dfapp_in.tail()
                dfapp_out = lEngine.forecast(dfapp_in, H);
                # print("Forecast Columns " , dfapp_out.columns);
                lForecast_DF[signal] = dfapp_out[signal]
                lForecast_DF[signal + '_Forecast'] = dfapp_out[signal + '_Forecast']
        print(lForecast_DF.columns);
        print(lForecast_DF.head());
        print(lForecast_DF.tail());
        return lForecast_DF;

    def computeTopDownHistoricalProportions(self, iForecast_DF):
        lLevelCount = len(self.mLevels);
        self.mAvgHistProp = {};
        self.mPropHistAvg = {};
        for level in  self.mStructure.keys():
            if(level > 0):
                for col in self.mStructure[level].keys():
                    self.mAvgHistProp[col] = {};
                    self.mPropHistAvg[col] = {};
                    for col1 in self.mStructure[level][col]:
                        self.mAvgHistProp[col][col1] = (iForecast_DF[col1] / iForecast_DF[col]).mean();
                        self.mPropHistAvg[col][col1] = iForecast_DF[col1].mean() / iForecast_DF[col].mean();
        print("AvgHitProp\n", self.mAvgHistProp);
        print("PropHistAvg\n", self.mPropHistAvg);
        pass
        
    def computeTopDownForecastedProportions(self, iForecast_DF):
        lLevelCount = len(self.mLevels);
        self.mForecastedProp = {};
        for level in  self.mStructure.keys():
            if(level > 0):
                for col in self.mStructure[level].keys():
                    self.mForecastedProp[col] = {};
                    for col1 in self.mStructure[level][col]:
                        self.mForecastedProp[col][col1] = (iForecast_DF[col1] / iForecast_DF[col]).mean();
        print("ForecastedProp\n", self.mForecastedProp);
        pass
        

    def computeBottomUpForecasts(self, iForecast_DF):
        lForecast_DF_BU = iForecast_DF.copy();
        print("STRUCTURE " , self.mStructure.keys());
        for level in self.mStructure.keys():
            for signal in self.mStructure[level].keys():
                new_BU_forecast = None;
                for col1 in self.mStructure[level][signal]:
                    if(new_BU_forecast is None):
                        new_BU_forecast = iForecast_DF[col1 + "_Forecast"];
                    else:
                        new_BU_forecast = new_BU_forecast + iForecast_DF[col1 + "_Forecast"];
                if(new_BU_forecast is None):
                    new_BU_forecast = iForecast_DF[signal + "_Forecast"];
                lForecast_DF_BU[signal + "_BU_Forecast"] = new_BU_forecast;
            
        print(lForecast_DF_BU.head());
        print(lForecast_DF_BU.tail());

        return lForecast_DF_BU;


    def reportOnBottomUpForecasts(self, iForecast_DF):
        lForecast_DF_BU = iForecast_DF;
        lPerfs = {};
        print("STRUCTURE " , self.mStructure.keys());
        for level in self.mStructure.keys():
            print("STRUCTURE_LEVEL " , level, self.mStructure[level].keys());
            print("MODEL_LEVEL " , level, self.mModels[level].keys());
            for signal in self.mStructure[level].keys():
                lEngine = self.mModels[level][signal];
                lPerf = lEngine.computePerf(lForecast_DF_BU[signal], lForecast_DF_BU[signal + "_Forecast"], signal)
                lPerf_BU = lEngine.computePerf(lForecast_DF_BU[signal], lForecast_DF_BU[signal + "_BU_Forecast"],  signal + "_BU")
                lPerfs[signal] = (lPerf , lPerf_BU);
            
        for (sig , perf) in lPerfs.items():
            print("PERF_REPORT_BU" , sig , perf[0].mL2,  perf[0].mMAPE, perf[1].mL2,  perf[1].mMAPE)
        return lPerfs;


    def computeTopDownForecasts(self, iForecast_DF , iProp , iPrefix):
        lForecast_DF_TD = iForecast_DF.copy();
        lLevelCount = len(self.mLevels);
        for level in sorted(self.mStructure.keys(), reverse=True):
            for signal in self.mStructure[level].keys():
                for col in self.mStructure[level][signal]:
                    new_TD_forecast = iForecast_DF[signal + "_Forecast"] * iProp[signal][col];
                    lForecast_DF_TD[col +"_" + iPrefix + "_TD_Forecast"] = new_TD_forecast;            
        print(lForecast_DF_TD.head());
        print(lForecast_DF_TD.tail());

        return lForecast_DF_TD;


    def computeOptimalCombination(self, iForecast_DF):
        lBaseNames = [];
        lLevelCount = len(self.mLevels);
        for level in  self.mStructure.keys():
            for col in self.mStructure[level].keys():
                lBaseNames.append(col);
        lBaseForecasts = iForecast_DF[lBaseNames];
        # TODO : use linalg.solve here
        S = self.mSummingMatrix;
        print(S.shape);
        lInv = np.linalg.inv(S.T.dot(S))
        lOptimalForecasts = S.dot(lInv).dot(S.T).dot(lBaseForecasts.values.T)
        print(lBaseForecasts.shape);
        print(lOptimalForecasts.shape);
        lOptimalNames = [(col + "_OC_Forecast") for col in lBaseNames];
        df = pd.DataFrame(lOptimalForecasts.T);
        df.columns = lOptimalNames;
        lForecast_DF_OC = pd.concat([iForecast_DF , df] , axis = 1);
        
        print(lForecast_DF_OC.head());
        print(lForecast_DF_OC.tail());
        return lForecast_DF_OC;

    def forecast(self , iInputDS, iHorizon):
        lAllLevelsDataset = self.create_all_levels_dataset(iInputDS);
        lForecast_DF = self.forecastAllModels(lAllLevelsDataset, iHorizon, self.mDateColumn);
        self.computeTopDownHistoricalProportions(lForecast_DF);
        lForecast_DF_BU = self.computeBottomUpForecasts(lForecast_DF);
        self.reportOnBottomUpForecasts(lForecast_DF_BU);
        lForecast_DF_TD_AHP = self.computeTopDownForecasts(lForecast_DF_BU, self.mAvgHistProp, "AHP") 
        lForecast_DF_TD_PHA = self.computeTopDownForecasts(lForecast_DF_TD_AHP, self.mPropHistAvg, "PHA")
        lForecast_DF_TD_PHA.to_csv("outputs/aggregated_forecasts_bu_td.csv");
        lForecast_DF_OC = self.computeOptimalCombination(lForecast_DF_TD_PHA);
        lForecast_DF_OC.to_csv("outputs/aggregated_forecasts_bu_td_oc.csv");
        return lForecast_DF_OC;
