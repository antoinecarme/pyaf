import pyaf.Bench.TS_datasets as tsds

import warnings
import pyaf.Bench.MComp as mcomp


tester1 = mcomp.cMComp_Tester(tsds.load_M2_comp() , "M2_COMP");

with warnings.catch_warnings():
    warnings.simplefilter("error")
    tester1.testSignals('BIGCAT')
    # tester2.testAllSignals()
    # tester2.run_multiprocessed();
