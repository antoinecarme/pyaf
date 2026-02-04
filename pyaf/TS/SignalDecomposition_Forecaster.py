#     #####             ####   ######       PyAF
#     ##  ##  ##   ##  ##  ##  ##           Python Automatic Forecasting
#     #####    ## ##   ######  ####   
#     ##        ##     ##  ##  ##           Version 5.x
#     ##       ##      ##  ##  ##           https://github.com/antoinecarme/pyaf
#             ##
# SPDX-FileCopyrightText: Copyright (c) (2017-) Antoine CARME <Antoine.Carme@outlook.com>
# SPDX-License-Identifier: BSD-3-Clause ( https://spdx.org/licenses/BSD-3-Clause.html )


from . import MissingData as tsmiss
from . import Options as tsopts
from . import Utils as tsutil


def forecast_one_signal(arg):
    (lSignal, iDecomposition, iInputDS, iHorizon) = arg
    lTimer = tsutil.cTimer(("MODEL_FORECAST_ONE_SIGNAL", {"Signal" : [lSignal], "Decomposition" : iDecomposition, "Horizon" : iHorizon }))
    # print(("MODEL_FORECAST_ONE_SIGNAL", {"Signal" : [lSignal], "Decomposition" : iDecomposition, "Horizon" : iHorizon }))
    lBestModel = iDecomposition.mBestModels[lSignal]
    lMissingImputer = tsmiss.cMissingDataImputer()
    lMissingImputer.mOptions = iDecomposition.mOptions
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
    
    def forecast(self, iDecomposition, iInputDS, iHorizons):
        lOptions = iDecomposition.mOptions
        lHorizons = {}
        for sig in iDecomposition.mSignals:
            if(dict == type(iHorizons)):
                lHorizons[sig] = iHorizons[sig]
            else:
                lHorizons[sig] = int(iHorizons)
        
        lInputDS = sample_signal_if_needed(iInputDS, lOptions)
        
        lForecastFrame = None
        args = [];
        for lSignal in iDecomposition.mSignals:
            args = args + [(lSignal, iDecomposition, iInputDS, lHorizons[lSignal])]

        NCores = min(len(args) , iDecomposition.mOptions.mNbCores) 
        lTimer = tsutil.cTimer(("MODEL_FORECAST",
                                {"Signals" : [lSignal for lSignal in iDecomposition.mSignals],
                                 "Cores" : NCores}))
        if(iDecomposition.mOptions.mParallelMode and  NCores > 1):
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=NCores) as executor:
                future_to_arg = {executor.submit(forecast_one_signal, arg): arg for arg in args}
                for future in concurrent.futures.as_completed(future_to_arg):
                    (lTimeInfo, lForecastFrame_i) = future.result()
                    lForecastFrame = self.merge_frames(lForecastFrame, lForecastFrame_i, lTimeInfo)
                    del lForecastFrame_i
        else:
            for arg in args:
                res = forecast_one_signal(arg)
                (lTimeInfo, lForecastFrame_i) = res
                lForecastFrame = self.merge_frames(lForecastFrame, lForecastFrame_i, lTimeInfo)
                del lForecastFrame_i
                
        return lForecastFrame;
