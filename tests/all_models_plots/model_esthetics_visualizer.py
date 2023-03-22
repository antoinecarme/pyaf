from __future__ import absolute_import

import pandas as pd
import numpy as np

import pyaf.ForecastEngine as autof
import pyaf.TS.Options as tsopts


def draw(iEngine, iFormula, iSignal):
    print("DRAWING_PREDICTION_INTERVALS", iFormula)
    b64_string = iEngine.mSignalDecomposition.mBestModels[iSignal].getPredictionIntervalPlot()
    import io, base64, imageio
    sourceString = io.BytesIO(base64.b64decode(b64_string))
    lArray = imageio.imread(sourceString, pilmode='RGB')
    return lArray

def plot_model(arg):
    (lDataset , lSpec) = arg
    lSignal = lDataset.mSignalVar
    print("PLOT_MODEL_START" , lSpec)
    lEngine = autof.cForecastEngine()
    lEngine
    H = lDataset.mHorizon;
    # lEngine.mOptions.enable_slow_mode();
    # lEngine.mOptions.mDebugPerformance = True;
    lEngine.mOptions.mNbCores = 1
    lEngine.mOptions.set_active_transformations(lSpec["Transformation"]);
    lEngine.mOptions.set_active_trends(lSpec["Trend"]);
    lEngine.mOptions.set_active_periodics(lSpec["Periodics"]);
    lEngine.mOptions.set_active_autoregressions(lSpec["AutoRegression"]);
    lEngine.mOptions.set_active_decomposition_types(lSpec["Decomposition"]);
    
    lPlots = {}
    lDict = {}
    try:
        lEngine.train(lDataset.mPastData , lDataset.mTimeVar , lDataset.mSignalVar, H);
        lEngine.getModelInfo();
        lDict = lEngine.to_dict()
        lMAPE = lDict[lSignal]["Model_Performance"]["MAPE"]
        lFormula = lDict[lSignal]["Model"]["Best_Decomposition"]
        print("PLOT_MODEL_END" , lSpec ,
              lFormula, lMAPE)
        img = draw(lEngine, lFormula, lSignal)
    except Exception as e:
        print("PLOT_MODEL_END_FAILED" , lSpec, str(e))
        return (lSpec, None, None, None)
        
    return (lSpec, img, lFormula, lMAPE)


class cModelEstheticsVisualizer:

    def __init__(self):
        self.mDataset = None
        self.mNbThreads = 12
        pass


    def gen_all(self):
        lEngine = autof.cForecastEngine()
        lSpecs = []
        lSpec = {}
        lKnownAutoRegressions = [x for x in tsopts.cModelControl.gKnownAutoRegressions if not x.endswith('X')]
        lKnownAutoRegressions = [x for x in lKnownAutoRegressions if (x != 'CROSTON')]
        lKnownPeriodics = ['NoCycle', 'BestCycle', 'Seasonal_MonthOfYear'];
        for tr in tsopts.cModelControl.gKnownTransformations:
            lSpec["Transformation"] = tr
            for tr1 in tsopts.cModelControl.gKnownTrends:
                lSpec["Trend"] = tr1
                for per in lKnownPeriodics:
                    lSpec["Periodics"] = per
                    for ar in lKnownAutoRegressions:
                        lSpec["AutoRegression"] = ar
                        for dec in tsopts.cModelControl.gKnownDecompositionTypes:
                            lSpec["Decomposition"] = dec
                            lSpecs = lSpecs + [(self.mDataset, lSpec.copy())]
        print("TESTED_MODELS" , len(lSpecs))
        lPlots = {}
        from multiprocessing import Pool
        pool = Pool(self.mNbThreads)    
        for res in pool.imap(plot_model, lSpecs):
            (lSpec, img, lFormula, lMAPE) = res
            lPlots[str(lSpec)] = (lSpec, img, lFormula, lMAPE)
        pool.close()
        pool.join()
        return lPlots

    def generate_video(self, iDataset):
        self.mDataset = iDataset
        lSignal = self.mDataset.mSignalVar
        plots = self.gen_all()
        images = []
        for(lSpec , lValue) in plots.items():
            (lSpec1, img, lFormula, lMAPE) = lValue
            images.append((img , lMAPE))

        images = sorted(images, key = lambda x : -x[1])

        import imageio as iio
        writer = iio.get_writer(lSignal + "_models_H_" + str(self.mDataset.mHorizon)+ ".mp4", fps=20)
        for im in images:
            writer.append_data(im[0])
        writer.close()
