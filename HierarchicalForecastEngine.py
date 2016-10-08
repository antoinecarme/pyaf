
import pandas as pd
import numpy as np

import AutoForecast.ForecastEngine as autof
from .TS import Options as tsopts
from .TS import Perf as tsperf
from .TS import Utils as tsutil
from .TS import SignalHierarchy as sighier

from .CodeGen import TS_CodeGen_Objects as tscodegen

class cHierarchicalForecastEngine:
        
    def __init__(self):
        self.mOptions = tsopts.cSignalDecomposition_Options();
        self.mSignalHierarchy = None;
        pass
    
    def train(self , iInputDS, iTime, iSignal, iHorizon, iHierarchy = None, iExogenousData = None, ):
        try:
            self.train_HierarchicalModel(iInputDS, iTime, iSignal, iHorizon, iHierarchy, iExogenousData);
        except tsutil.ForecastError as error:
            print('caught this training error: ' + repr(error))            
            raise Exception("HIERARCHICAL_TRAIN_FAILED");
        pass

    def forecast(self , iInputDS, iHorizon):
        try:
            lForecastFrame = self.forecast_HierarchicalModel(iInputDS, iHorizon);
            return lForecastFrame;
        except tsutil.ForecastError as error:
            print('caught this forecast error: ' + repr(error))
            raise Exception("HIERARCHICAL_FORECAST_FAILED");
        
    def getModelInfo(self):
        self.mSignalHierarchy.getModelInfo();

    
    def standrdPlots(self , name = None):
        self.mSignalHierarchy.standrdPlots(name);

    def getPlotsAsDict(self):
        return self.mSignalDecomposition.getPlotsAsDict();

    def to_json(self):
        return self.mSignalDecomposition.to_json();

    def computePerf(self, actual, predicted , name):
        lForecastPerf =  tsperf.cPerf();
        lForecastPerf.compute(actual, predicted, name);
        return lForecastPerf;

    def generateCode(self, iDSN = None, iDialect = None):
        lCodeGenerator = tscodegen.cDecompositionCodeGenObject(iDSN, iDialect);
        lSQL = lCodeGenerator.generateCode(self);
        # print("GENERATED_SQL_CODE" , lSQL);
        return lSQL;


    def train_HierarchicalModel(self , iInputDS, iTime, iSignal, iHorizon, iHierarchy, iExogenousData = None):
        self.mSignalHierarchy = sighier.cSignalHierarchy();
        self.mSignalHierarchy.mHierarchy = iHierarchy;
        self.mSignalHierarchy.mDateColumn = iTime;
        self.mSignalHierarchy.mHorizon = iHorizon;
        self.mSignalHierarchy.mExogenousData = iExogenousData;        
        self.mSignalHierarchy.mTrainingDataset = iInputDS;        
        self.mSignalHierarchy.mOptions = self.mOptions;        
        self.mSignalHierarchy.fit();


    def forecast_HierarchicalModel(self , iInputDS, iHorizon):
        return self.mSignalHierarchy.forecast(iInputDS, iHorizon);
