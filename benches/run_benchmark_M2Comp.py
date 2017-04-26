import pandas as pd
import numpy as np
import pyaf.Bench.MComp as mcomp
import pyaf.Bench.TS_datasets as tsds
import warnings


tester2 = mcomp.cMComp_Tester(tsds.load_M2_comp() , "M2_COMP");

with warnings.catch_warnings():
    warnings.simplefilter("error")
    # tester1.testSignal('')
    # tester2.testAllSignals()
    tester2.run_multiprocessed();
