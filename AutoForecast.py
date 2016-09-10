from SignalDecomposition import SignalDecomposition as tsdec
from SignalDecomposition import SignalDecomposition_Options as tsopts
from SignalDecomposition import SignalDecomposition_Perf as tsperf
from SignalDecomposition import SignalDecomposition_utils as tsutil

import TS_CodeGen_Objects as tscodegen

class cAutoForecast:
        
    def __init__(self):
        self.mSignalDecomposition = tsdec.cSignalDecomposition();
        self.mOptions = tsopts.cSignalDecomposition_Options();
        pass

    
    def train(self , iInputDS, iTime, iSignal, iHorizon, iExogenous = []):
        try:
            self.mSignalDecomposition.mOptions = self.mOptions;
            self.mSignalDecomposition.train(iInputDS, iTime, iSignal, iHorizon, iExogenous);
        except tsutil.ForecastError as error:
            print('caught this training error: ' + repr(error))
            print("AUTO_FORECAST_FAILED" , name)
            raise;
        pass

    def forecast(self , iInputDS, iHorizon):
        try:
            lForecastFrame = self.mSignalDecomposition.forecast(iInputDS, iHorizon);
            return lForecastFrame;
        except tsutil.ForecastError as error:
            print('caught this forecast error: ' + repr(error))
            print("AUTO_FORECAST_FAILED" , name)
            raise;
        
    def getModelInfo(self):
        return  self.mSignalDecomposition.getModelInfo();

    
    def standrdPlots(self , name = None):
        self.mSignalDecomposition.standrdPlots(name);

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
