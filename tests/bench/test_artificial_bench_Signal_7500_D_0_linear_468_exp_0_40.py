import pyaf.Bench.TS_datasets as tsds

import pyaf.Bench.Artificial as art
import warnings


H = 12;
lSignal = "Signal_7500_D_0_linear_468_exp_0_40"


with warnings.catch_warnings():
    warnings.simplefilter("error")
    tester1 = art.cArtificial_Tester(tsds.load_artificial_datsets("XL", iName = lSignal) , "ARTIFICIAL_XL");
    tester1.testSignals(lSignal , H);
