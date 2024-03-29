# Copyright (C) 2016 Antoine Carme <Antoine.Carme@outlook.com>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np

from .TS import Options as tsopts
from .TS import Utils as tsutil


class cHierarchicalForecastEngine:
        
    def __init__(self):
        self.mOptions = tsopts.cSignalDecomposition_Options();
        self.mSignalHierarchy = None;
        pass
    
    def train(self , iInputDS, iTime, iSignal, iHorizon, iHierarchy = None, iExogenousData = None, ):
        try:
            self.train_HierarchicalModel(iInputDS, iTime, iSignal, iHorizon, iHierarchy, iExogenousData);
        except tsutil.PyAF_Error as error:
            raise error
        except Exception as error:
            # print('caught this training error: ' + repr(error))            
            raise tsutil.PyAF_Error("HIERARCHICAL_TRAIN_FAILED");
        pass

    def forecast(self , iInputDS, iHorizon):
        try:
            lForecastFrame = self.forecast_HierarchicalModel(iInputDS, iHorizon);
            return lForecastFrame;
        except tsutil.PyAF_Error as error:
            raise error
        except Exception as error:
            # print('caught this forecast error: ' + repr(error))
            raise tsutil.PyAF_Error("HIERARCHICAL_FORECAST_FAILED");
        
    def getModelInfo(self):
        self.mSignalHierarchy.getModelInfo();

    
    def standardPlots(self , name = None):
        self.mSignalHierarchy.standardPlots(name);

    def getPlotsAsDict(self):
        return self.mSignalHierarchy.getPlotsAsDict();

    def to_dict(self):
        return self.mSignalHierarchy.to_dict();

    def to_json(self):
        lDict = self.to_dict()
        import json
        return json.dumps(lDict, default = lambda o: o.__dict__, indent=4, sort_keys=True);

    def computePerf(self, actual, predicted , name):
        from .TS import Perf as tsperf
        lForecastPerf =  tsperf.cPerf();
        lForecastPerf.compute(actual, predicted, name);
        return lForecastPerf;

    def create_signal_hierarchy(self , iInputDS, iTime, iSignal, iHorizon, iHierarchy, iExogenousData = None):
        lSignalHierarchy = None;
        if(iHierarchy['Type'] == "Grouped"):
            from .TS import Signal_Grouping as siggroup
            lSignalHierarchy = siggroup.cSignalGrouping();
        elif(iHierarchy['Type'] == "Temporal"):
            from .TS import Temporal_Hierarchy as temphier
            lSignalHierarchy = temphier.cTemporalHierarchy();
        else:
            from .TS import SignalHierarchy as sighier
            lSignalHierarchy = sighier.cSignalHierarchy();
            
        lSignalHierarchy.mHierarchy = iHierarchy;
        lSignalHierarchy.mDateColumn = iTime;
        lSignalHierarchy.mSignal = iSignal;
        lSignalHierarchy.mHorizon = iHorizon;
        lSignalHierarchy.mExogenousData = iExogenousData;        
        lSignalHierarchy.mTrainingDataset = iInputDS;        
        lSignalHierarchy.mOptions = self.mOptions;        
        return lSignalHierarchy;


    def plot_Hierarchy(self , iInputDS, iTime, iSignal, iHorizon, iHierarchy, iExogenousData = None):
        lSignalHierarchy = self.create_signal_hierarchy(iInputDS, iTime, iSignal, iHorizon, iHierarchy, iExogenousData);
        lSignalHierarchy.create_HierarchicalStructure();
        lSignalHierarchy.plot();
        return lSignalHierarchy;

    def train_HierarchicalModel(self , iInputDS, iTime, iSignal, iHorizon, iHierarchy, iExogenousData = None):
        lSignalHierarchy = self.create_signal_hierarchy(iInputDS, iTime, iSignal, iHorizon, iHierarchy, iExogenousData);
        self.mSignalHierarchy = lSignalHierarchy;        
        self.mSignalHierarchy.fit();


    def forecast_HierarchicalModel(self , iInputDS, iHorizon):
        return self.mSignalHierarchy.forecast(iInputDS, iHorizon);
