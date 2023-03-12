import pyaf.Bench.TS_datasets as tsds

import pyaf.Bench.Artificial as art
import warnings


lSignal = "Signal_1000_H_0_LinearTrend_0_exp_1.0_10"


with warnings.catch_warnings():
    warnings.simplefilter("error")
    tester1 = art.cArtificial_Tester(tsds.load_artificial_datsets("L", iName = lSignal) , "ARTIFICIAL_L");
    tester1.testSignals(lSignal);
