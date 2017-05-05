from __future__ import absolute_import

import pandas as pd
import numpy as np


import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds


b1 = tsds.load_ozone()
df = b1.mPastData


lEngine = autof.cForecastEngine()
lEngine

H = b1.mHorizon;
lEngine.train(df , b1.mTimeVar , b1.mSignalVar, H);
lEngine.getModelInfo();

lDict = lEngine.mSignalDecomposition.mBestModel.mTrainingVersionInfo
for k in sorted(lDict.keys()):
    print( k , lDict[k]);
