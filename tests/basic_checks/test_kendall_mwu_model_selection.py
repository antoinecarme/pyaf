import numpy as np
import pandas as pd

import pyaf.ForecastEngine as autof

N = 80
df = pd.DataFrame({
    "Date" : pd.date_range("2000-01-01", periods = N, freq = "D"),
    "Signal" : 10.0 + np.arange(N) + np.sin(np.arange(N) / 3.0)
})

for lCriterion in ["MWU", "KendallTau"]:
    lEngine = autof.cForecastEngine()
    lEngine.mOptions.mModelSelection_Criterion = lCriterion
    lEngine.train(df, "Date", "Signal", 5)
    assert(lEngine.mSignalDecomposition.mBestModel is not None)
