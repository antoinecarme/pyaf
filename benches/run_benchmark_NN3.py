import pandas as pd
import numpy as np
import pyaf.Bench.TS_datasets as tsds
import pyaf.Bench.NN3 as tNN3
import warnings

#%matplotlib inline

tester1 = tNN3.cNN_Tester(tsds.load_NN3_part1() , "NN3_PART_1");
with warnings.catch_warnings():
    warnings.simplefilter("error")
    #tester1.testSignal('NN3-059')
    # tester1.testAllSignals()
    tester1.run_multiprocessed();

tester2 = tNN3.cNN_Tester(tsds.load_NN3_part2() , "NN3_PART_2");
with warnings.catch_warnings():
    warnings.simplefilter("error")
    # tester2.testSignal('NN3_103')
    # tester2.testAllSignals()
    tester2.run_multiprocessed();
