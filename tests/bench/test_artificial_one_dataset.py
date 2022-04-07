import pyaf.Bench.TS_datasets as tsds

import pyaf.Bench.Artificial as art
import warnings


H = 2;
lSignal = "Signal_100_D_0_linear_12_exp_4_40"


with warnings.catch_warnings():
    warnings.simplefilter("error")
    tester1 = art.cArtificial_Tester(tsds.load_artificial_datsets("S", iName = lSignal) , "ARTIFICIAL_S");
    # lSignal = list(tester1.mTSSpecPerSignal.keys())[0];
    tester1.testSignals(lSignal , H);
    # tester1.run_multiprocessed(18);
    del tester1;
    # tester2 = art.cArtificial_Tester(tsds.load_artificial_datsets("M") , "ARTIFICIAL_M");
    # lSignal = list(tester2.mTSSpecPerSignal.keys())[0];
    # tester2.testSignals(lSignal , H);
    #tester2.testSignals("Signal_450_D_0_poly_105_exp_4_90" , H);
    # tester2.run_multiprocessed(18);
    # del tester2;
    # tester3 = art.cArtificial_Tester(tsds.load_artificial_datsets("L") , "ARTIFICIAL_L");
    # lSignal = list(tester3.mTSSpecPerSignal.keys())[0];
    # tester3.testSignals(lSignal , H);
    # tester3.testSignals("Signal_700_D_0_poly_15_exp_0", H);
    # tester3.run_multiprocessed(18);
    # del tester3;
    # tester4 = art.cArtificial_Tester(tsds.load_artificial_datsets("XL") , "ARTIFICIAL_XL");
    # lSignal = list(tester4.mTSSpecPerSignal.keys())[0];
    # tester4.testSignals(lSignal , H);
    # tester4.testSignals("Signal_3000_D_0_poly_60_exp_2", H);
    # tester4.run_multiprocessed(18);
    # del tester4;
