# Copyright (C) 2016 Antoine Carme <Antoine.Carme@outlook.com>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

from . import Options as tsopts
from . import Utils as tsutil
        
class cSignalDecomposition:
        
    def __init__(self):
        self.mSigDecBySplitAndTransform = {};
        self.mOptions = tsopts.cSignalDecomposition_Options();
        self.mExogenousData = None;
        pass

    def checkData(self, iInputDS, iTime, iSignal, iHorizon, iExogenousData):        
        if(iHorizon != int(iHorizon)):
            raise tsutil.PyAF_Error("PYAF_ERROR_NON_INTEGER_HORIZON " + str(iHorizon));
        if(iHorizon < 1):
            raise tsutil.PyAF_Error("PYAF_ERROR_NEGATIVE_OR_NULL_HORIZON " + str(iHorizon));
        if(iTime not in iInputDS.columns):
            raise tsutil.PyAF_Error("PYAF_ERROR_TIME_COLUMN_NOT_FOUND " + str(iTime));
        for lSignal in [iSignal]:
            if(lSignal not in iInputDS.columns):
                raise tsutil.PyAF_Error("PYAF_ERROR_SIGNAL_COLUMN_NOT_FOUND " + str(lSignal));
            type2 = iInputDS[lSignal].dtype
            # print(type2)
            if(type2.kind != 'i' and type2.kind != 'u' and type2.kind != 'f'):
                raise tsutil.PyAF_Error("PYAF_ERROR_SIGNAL_COLUMN_TYPE_NOT_ALLOWED '" + str(lSignal) + "' '" + str(type2) + "'");
        type1 = iInputDS[iTime].dtype
        # print(type1)
        if(type1.kind != 'M' and type1.kind != 'i' and type1.kind != 'u' and type1.kind != 'f'):
            raise tsutil.PyAF_Error("PYAF_ERROR_TIME_COLUMN_TYPE_NOT_ALLOWED '" + str(iTime) + "' '" + str(type1) + "'");
        # time in exogenous data should be the strictly same type as time in training dataset (join needed)
        if(iExogenousData is not None):
            lExogenousDataFrame = iExogenousData[0];
            lExogenousVariables = iExogenousData[1];
            if(iTime not in lExogenousDataFrame.columns):
                raise tsutil.PyAF_Error("PYAF_ERROR_TIME_COLUMN_NOT_FOUND_IN_EXOGENOUS " + str(iTime));
            for exog in lExogenousVariables:
                if(exog not in lExogenousDataFrame.columns):
                    raise tsutil.PyAF_Error("PYAF_ERROR_EXOGENOUS_VARIABLE_NOT_FOUND " + str(exog));
                
            type3 = lExogenousDataFrame[iTime].dtype
            if(type1 != type3):
                raise tsutil.PyAF_Error("PYAF_ERROR_INCOMPATIBLE_TIME_COLUMN_TYPE_IN_EXOGENOUS '" + str(iTime) + "' '" + str(type1)  + "' '" + str(type3) + "'");

    def reinterpret_by_signal_args(self, iTimes, iSignals, iHorizons, iExogenousData):
        # backward compatibility
        self.mSignals = iSignals
        if(str == type(iSignals)):
            self.mSignals = [iSignals]
        self.mDateColumns = {}
        for sig in self.mSignals:
            if(dict == type(iTimes)):
                self.mDateColumns[sig] = iTimes[sig]
            else:
                self.mDateColumns[sig] = iTimes
        self.mHorizons = {}
        for sig in self.mSignals:
            if(dict == type(iHorizons)):
                self.mHorizons[sig] = iHorizons[sig]
            else:
                self.mHorizons[sig] = int(iHorizons)
        self.mExogenousData = {}
        for sig in self.mSignals:
            if(dict == type(iExogenousData)):
                self.mExogenousData[sig] = iExogenousData[sig]
            else:
                self.mExogenousData[sig] = iExogenousData
        
            
    def train(self , iInputDS, iTimes, iSignals, iHorizons, iExogenousData = None):
        from . import SignalDecomposition_Trainer as tstrainer

        self.reinterpret_by_signal_args(iTimes, iSignals, iHorizons, iExogenousData)
        # print(iInputDS.shape, iInputDS.columns, self.mSignals, self.mDateColumns, self.mHorizons)
        lTimer = tsutil.cTimer(("TRAINING", {"Signals" : self.mSignals, "Horizons" : self.mHorizons}))

        for sig in self.mSignals:
            self.checkData(iInputDS, self.mDateColumns[sig], sig, self.mHorizons[sig], self.mExogenousData[sig]);

        self.mTrainingDataset = iInputDS; 

        lTrainer = tstrainer.cSignalDecompositionTrainer()
        lSplits = lTrainer.define_splits()
        lTrainer.mOptions = self.mOptions;
        lTrainer.mExogenousData = self.mExogenousData;
        
        lTrainer.train(iInputDS, lSplits , self.mDateColumns, self.mSignals, self.mHorizons)
        self.mBestModels = lTrainer.mBestModels
        self.mTrPerfDetailsBySignal = lTrainer.mTrPerfDetails
        self.mModelShortListBySignal = lTrainer.mModelShortList
        # some backward compatibility
        lFirstSignal = self.mSignals[0] 
        self.mBestModel = self.mBestModels[lFirstSignal]
        self.mTrainingTime = lTimer.get_elapsed_time()
        for (lSignal, lBestModel) in self.mBestModels.items():
            lBestModel.clean_dataframes()
        del lTrainer
        


    def forecast(self , iInputDS, iHorizon):
        from . import SignalDecomposition_Forecaster as tsforec
        lTimer = tsutil.cTimer(("FORECASTING", {"Signals" : self.mSignals, "Horizon" : iHorizon}))
        lForecaster = tsforec.cSignalDecompositionForecaster()
        lForecastFrame = lForecaster.forecast(self, iInputDS, iHorizon)
        del lForecaster
        
        return lForecastFrame;


    def getModelFormula(self):
        lFormula = {}
        for lSignal in self.mSignals:
            lFormula[lSignal] = self.mBestModel.getFormula();
        return lFormula;

    def get_competition_details(self):
        logger = tsutil.get_pyaf_logger();
        for lSignal in self.mSignals:
            logger.info("COMPETITION_DETAIL_START '" + lSignal + "'");
            lShortList_Dict = self.mModelShortListBySignal[lSignal].to_dict(orient = 'index')
            # print(lShortList_Dict)
            for k in sorted(lShortList_Dict.keys()):
                v = lShortList_Dict[k]
                logger.info("COMPETITION_DETAIL_SHORT_LIST '" + lSignal + "' " + str(k) + " " + str(v));
            logger.info("COMPETITION_DETAIL_END '" + lSignal + "'");

    def getModelInfo(self):
        for lSignal in self.mSignals:
            self.mBestModels[lSignal].getInfo()
        logger = tsutil.get_pyaf_logger();
        logger.info("TRAINING_TIME_IN_SECONDS " + str(self.mTrainingTime));
        self.get_competition_details()
        

    def to_dict(self, iWithOptions = False):
        dict1 = {}
        dict1["Training_Time"] = self.mTrainingTime
        for lSignal in self.mSignals:
            dict1[lSignal] = self.mBestModels[lSignal].to_dict(iWithOptions);
        
        return dict1
        
    def standardPlots(self, name = None, format = 'png'):
        lTimer = tsutil.cTimer(("PLOTTING", {"Signals" : self.mSignals}))
        for lSignal in self.mSignals:
            lName = name
            if(name is not None):
                lName = str(name) + "_" + str(lSignal)
            self.mBestModels[lSignal].standardPlots(lName, format);
        
        
    def getPlotsAsDict(self):
        lDict = {}
        for lSignal in self.mSignals:
            lDict[lSignal] = self.mBestModels[lSignal].getPlotsAsDict();
        
        return lDict;
