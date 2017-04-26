import warnings

import pyaf.Bench.TS_datasets as tsds

import pyaf.Bench.MComp as mcomp

tester7 = mcomp.cMComp_Tester(tsds.load_M4_comp("FINANCE") , "M4_COMP");
with warnings.catch_warnings():
    warnings.simplefilter("error")
    tester7.testSignals('FIN28')
    # tester7.testSignal('ECON0299')
    # tester7.testAllSignals()
    # tester7.run_multiprocessed(20);
