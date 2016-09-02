from SignalDecomposition import SignalDecomposition as tsdec
from SignalDecomposition import SignalDecomposition_Options as tsopts
from SignalDecomposition import SignalDecomposition_Perf as tsperf
from SignalDecomposition import SignalDecomposition_utils as tsutil

class cAutoForecast:
        
    def __init__(self):
        self.mSignalDecomposition = tsdec.cSignalDecomposition();
        self.mOptions = tsopts.cSignalDecomposition_Options();
        pass

    
    def train(self , iInputDS, iTime, iSignal, iHorizon):
        try:
            self.mSignalDecomposition.mOptions = self.mOptions;
            self.mSignalDecomposition.train(iInputDS, iTime, iSignal, iHorizon);
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

    
    def standrdPlots(self):
        self.mSignalDecomposition.standrdPlots();

    def to_json(self):
        self.mSignalDecomposition.to_json();

    def computePerf(self, actual, predicted , name):
        lForecastPerf =  tsperf.cPerf();
        lForecastPerf.compute(actual, predicted, name);
        return lForecastPerf;
