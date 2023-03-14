# Copyright (C) 2023 Antoine Carme <Antoine.Carme@outlook.com>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

from . import MissingData as tsmiss
from . import Options as tsopts
from . import Utils as tsutil


def forecast_one_signal(arg):
    (lSignal, iDecomsposition, iInputDS, iHorizon) = arg
    lBestModel = iDecomsposition.mBestModels[lSignal]
    lMissingImputer = tsmiss.cMissingDataImputer()
    lMissingImputer.mOptions = iDecomsposition.mOptions
    lInputDS = iInputDS[[lBestModel.mTime, lSignal]].copy()
    lInputDS = lMissingImputer.apply(lInputDS, lBestModel.mTime, lBestModel.mOriginalSignal)
    lForecastFrame_i = lBestModel.forecast(lInputDS, iHorizon);
    return (lBestModel.mTimeInfo, lForecastFrame_i)

def sample_signal_if_needed(iInputDS, iOptions):
    logger = tsutil.get_pyaf_logger();
    lInputDS = iInputDS
    if(iOptions.mActivateSampling):
        if(iOptions.mDebugProfile):
            logger.info("PYAF_MODEL_SAMPLING_ACTIVATED " +
                        str((iOptions.mSamplingThreshold, iOptions.mSeed)));
        lInputDS = iInputDS.tail(iOptions.mSamplingThreshold);
    return lInputDS

class cSignalDecompositionForecaster:

    def __init__(self):
        pass

    def merge_frames(self, iFullFrame, iOneSignalFrame, iTimeInfo):
        if(iFullFrame is None):
            return iOneSignalFrame
        lTime = iFullFrame.columns[0]
        lOneSignalCommonColumns = [col for col in iOneSignalFrame.columns if col in iFullFrame.columns]
        lOneSignalCommonColumns = [col for col in lOneSignalCommonColumns if col not in [lTime , iTimeInfo.mTime]]
        lOneSignalFrame = iOneSignalFrame.drop(lOneSignalCommonColumns, axis = 1)
        lForecastFrame = iFullFrame.merge(lOneSignalFrame, how='left', left_on=lTime, right_on=iTimeInfo.mTime);
        return lForecastFrame
    
    def forecast(self, iDecomsposition, iInputDS, iHorizons):
        lOptions = iDecomsposition.mOptions
        lHorizons = {}
        for sig in iDecomsposition.mSignals:
            if(dict == type(iHorizons)):
                lHorizons[sig] = iHorizons[sig]
            else:
                lHorizons[sig] = int(iHorizons)
        
        lInputDS = sample_signal_if_needed(iInputDS, lOptions)
        
        lForecastFrame = None
        args = [];
        for lSignal in iDecomsposition.mSignals:
            args = args + [(lSignal, iDecomsposition, iInputDS, lHorizons[lSignal])]

        NCores = min(len(args) , iDecomsposition.mOptions.mNbCores) 
        if(iDecomsposition.mOptions.mParallelMode and  NCores > 1):
            from multiprocessing import Pool
            pool = Pool(NCores)
        
            for res in pool.imap(forecast_one_signal, args):
                (lTimeInfo, lForecastFrame_i) = res
                lForecastFrame = self.merge_frames(lForecastFrame, lForecastFrame_i, lTimeInfo)
                del lForecastFrame_i
            pool.close()
            pool.join()
        else:
            for arg in args:
                res = forecast_one_signal(arg)
                (lTimeInfo, lForecastFrame_i) = res
                lForecastFrame = self.merge_frames(lForecastFrame, lForecastFrame_i, lTimeInfo)
                del lForecastFrame_i
                
        return lForecastFrame;
