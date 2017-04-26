import pyaf.Bench.TS_datasets as tsds

import pyaf.Bench.MComp as mcomp

import warnings

tester3 = mcomp.cMComp_Tester(tsds.load_M3_Y_comp() , "M3_Y_COMP");

with warnings.catch_warnings():
    warnings.simplefilter("error")
    tester3.testSignals('N366')
    # tester3.testAllSignals()
    # tester3.run_multiprocessed()

tester4 = mcomp.cMComp_Tester(tsds.load_M3_Q_comp() , "M3_Q_COMP");
with warnings.catch_warnings():
    warnings.simplefilter("error")
    # tester1.testSignal('')
    # tester4.testAllSignals()
    # tester4.run_multiprocessed()

tester5 = mcomp.cMComp_Tester(tsds.load_M3_M_comp() , "M3_M_COMP");
with warnings.catch_warnings():
    warnings.simplefilter("error")
    # tester1.testSignal('')
    # tester5.testAllSignals()
    # tester5.run_multiprocessed()

tester6 = mcomp.cMComp_Tester(tsds.load_M3_Other_comp() , "M3_OTHER_COMP");
with warnings.catch_warnings():
    warnings.simplefilter("error")
    # tester1.testSignal('')
    # tester6.testAllSignals()
    # tester6.run_multiprocessed()
