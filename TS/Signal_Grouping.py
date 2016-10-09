import pandas as pd
import numpy as np
import itertools


import AutoForecast.ForecastEngine as autof
from . import Options as tsopts
from . import Perf as tsperf
from . import Utils as tsutil

from CodeGen import TS_CodeGen_Objects as tscodegen


class cSignalGrouping:

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
        self.mModels = None;
        
    def info(self):
        lStr2 = ""
        return lStr2;


    def to_json(self):
        dict1 = {};
        return dict1;

    def add_level(self, previous_level):
        level = previous_level + 1;
        self.mStructure[level] = {};
        for group in self.mStructure[previous_level]:
            for k in range(len(group)):
                if(group[k] != ""):
                    new_group = list(group);
                    new_group[k] = "";
                    new_group = tuple(new_group);
                    if(new_group not in self.mStructure[level]):
                        self.mStructure[level][new_group] = set();
                    self.mStructure[level][new_group].add(group)
        print(self.mStructure[level]);

    def create_HierarchicalStructure(self):
        
        # lGroups = {};
        # lGroups["State"] = ["NSW","VIC","QLD","SA","WA","NT","ACT","TAS"];
        # lGroups["Gender"] = ["female","male"];
        # lHierarchy['GroupOrder']= ["State" , "Gender"];
        
        # self.mLevels = self.mHierarchy['Levels'];
        lGroups = self.mHierarchy['Groups']
        self.mStructure = {};
        array1 = [ lGroups[k] for k in self.mHierarchy['GroupOrder'] ];
        prod = itertools.product( *array1 );
        # prod = itertools.product(['a' , 'b'] , ['1' , '2'] , ['cc' , 'dd']);
        level = 0;
        self.mStructure[level] = {}
        for k in prod:
            print(k);
            self.mStructure[level][k] = set();
        while(len(self.mStructure[level]) > 1):
            self.add_level(level);
            level = level + 1;
        
        print(self.mStructure);
        pass

    def tuple_to_string(self, k):
        str1 = "_".join(list(k));
        print(k , "=>" , str1);
        return str1;
    
    def create_all_levels_dataset(self, df):
        lAllLevelsDataset = df.copy();
        i = 0;
        for k in self.mStructure[0]:
            print("MAPPING_ORIGINAL_COLUMN" , df.columns[i + 1], "=>" , self.tuple_to_string(k))
            lAllLevelsDataset[self.tuple_to_string(k)] = df[df.columns[i + 1]];
            i = i + 1;
        lLevelCount = len(self.mStructure);
        print(lAllLevelsDataset.columns);
        for level in range(lLevelCount):
            if(level > 0):
                for col in self.mStructure[level].keys():
                    col_label = self.tuple_to_string(col);
                    new_col = None;
                    for col1 in self.mStructure[level][col]:
                        col1_label = self.tuple_to_string(col1);
                        if(new_col is None):
                            new_col = lAllLevelsDataset[col1_label];
                        else:
                            new_col = new_col + lAllLevelsDataset[col1_label];
                    lAllLevelsDataset[col_label] = new_col;
        print(lAllLevelsDataset.columns);
        return lAllLevelsDataset;


    def addVars(self, df):
        lAllLevelsDataset = self.create_all_levels_dataset(df);
        return lAllLevelsDataset;

    def transformDataset(self, df):
        df = self.addVars(df);
        return df;


    def create_all_levels_models(self, iAllLevelsDataset, H, iDateColumn):
        self.mModels = {};
        lLevelCount = len(self.mStructure);
        for level in range(lLevelCount):
            self.mModels[level] = {};
            for signal in self.mStructure[level].keys():
                signal1 = self.tuple_to_string(signal);
                lEngine = autof.cForecastEngine()
                lEngine.mOptions = self.mOptions;
                lEngine.train(iAllLevelsDataset , iDateColumn , signal1, H);
                lEngine.getModelInfo();
                self.mModels[level][signal] = lEngine;
        pass


    def fit(self):
        self.create_HierarchicalStructure();
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
                signal1 = self.tuple_to_string(signal);
                lEngine = self.mModels[level][signal];
                lEngine.standrdPlots(name + "_Grouping_Level_Signal_" + str(level) + "_" + signal1);


    def forecastAllModels(self, iAllLevelsDataset, H, iDateColumn):
        lForecast_DF = pd.DataFrame();
        lForecast_DF[iDateColumn] = iAllLevelsDataset[iDateColumn]
        for level in self.mModels.keys():
            for signal in self.mModels[level].keys():
                signal1 = self.tuple_to_string(signal);
                lEngine = self.mModels[level][signal];
                dfapp_in = iAllLevelsDataset.copy();
                # dfapp_in.tail()
                dfapp_out = lEngine.forecast(dfapp_in, H);
                # print("Forecast Columns " , dfapp_out.columns);
                lForecast_DF[signal1] = dfapp_out[signal1]
                lForecast_DF[signal1 + '_Forecast'] = dfapp_out[signal1 + '_Forecast']
        print(lForecast_DF.columns);
        print(lForecast_DF.head());
        print(lForecast_DF.tail());
        return lForecast_DF;


    def reportOnBottomUpForecasts(self, iForecast_DF):
        lForecast_DF_BU = pd.DataFrame();
        lDateColumn = self.mDateColumn;
        lForecast_DF_BU[ lDateColumn ] = iForecast_DF[ lDateColumn ];
        lPerfs = {};
        for level in self.mStructure.keys():
            for signal in self.mStructure[level].keys():
                signal1 = self.tuple_to_string(signal);
                lEngine = self.mModels[level][signal];
                new_BU_forecast = None;
                for col1 in self.mStructure[level][signal]:
                    signal2 = self.tuple_to_string(col1);
                    if(new_BU_forecast is None):
                        new_BU_forecast = iForecast_DF[signal2 + "_Forecast"];
                    else:
                        new_BU_forecast = new_BU_forecast + iForecast_DF[signal2 + "_Forecast"];
                lForecast_DF_BU[signal1] = iForecast_DF[signal1];            
                lForecast_DF_BU[signal1 + "_Forecast"] = iForecast_DF[signal1 + "_Forecast"];
                if(new_BU_forecast is None):
                    new_BU_forecast = iForecast_DF[signal1 + "_Forecast"];
                lForecast_DF_BU[signal1 + "_BU_Forecast"] = new_BU_forecast;
                lPerf = lEngine.computePerf(lForecast_DF_BU[signal1], lForecast_DF_BU[signal1 + "_Forecast"], signal1)
                lPerf_BU = lEngine.computePerf(lForecast_DF_BU[signal1], lForecast_DF_BU[signal1 + "_BU_Forecast"],  signal1 + "_BU")
                lPerfs[signal1] = (lPerf , lPerf_BU);
            
        print(lForecast_DF_BU.head());
        print(lForecast_DF_BU.tail());

        for (sig , perf) in lPerfs.items():
            print("PERF_REPORT_BU" , sig , perf[0].mL2,  perf[0].mMAPE, perf[1].mL2,  perf[1].mMAPE)
        return lForecast_DF_BU;

    def forecast(self , iInputDS, iHorizon):
        lAllLevelsDataset = self.create_all_levels_dataset(iInputDS);
        lForecast_DF = self.forecastAllModels(lAllLevelsDataset, iHorizon, self.mDateColumn);
        self.reportOnBottomUpForecasts(lForecast_DF);
        return lForecast_DF;
