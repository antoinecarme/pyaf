import pandas as pd
import numpy as np
import Bench.TS_datasets as tsds
import Bench.Artificial as art
import warnings


H = 2;

with warnings.catch_warnings():
    warnings.simplefilter("error")
    tester1 = art.cArtificial_Tester(tsds.load_artificial_datsets("S") , "ARTIFICIAL_S");
    tester1.testSignals("Signal_5_D_0_poly_30__0" , H);
    # tester1.run_multiprocessed(18);
    del tester1;
    tester2 = art.cArtificial_Tester(tsds.load_artificial_datsets("M") , "ARTIFICIAL_M");
    tester2.testSignals("Signal_400_D_0_linear_0__2" , H);
    # tester2.run_multiprocessed(18);
    del tester2;
    tester3 = art.cArtificial_Tester(tsds.load_artificial_datsets("L") , "ARTIFICIAL_L");
    tester3.testSignals("Signal_700_D_0_poly_15_exp_0", H);
    # tester3.run_multiprocessed(18);
    del tester3;
    tester4 = art.cArtificial_Tester(tsds.load_artificial_datsets("XL") , "ARTIFICIAL_XL");
    tester4.testSignals("Signal_3000_D_0_poly_60_exp_2", H);
    # tester4.run_multiprocessed(18);
    del tester4;
