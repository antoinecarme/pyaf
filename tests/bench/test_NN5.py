import pyaf.Bench.TS_datasets as tsds

import pyaf.Bench.NN3 as tNN3
import warnings

#%matplotlib inline

tester1 = tNN3.cNN_Tester(tsds.load_NN5() , "NN5");
with warnings.catch_warnings():
    warnings.simplefilter("error")
    tester1.testSignals('NN5-025')
    # tester1.testAllSignals()
    # tester1.run_multiprocessed()
