import pandas as pd
import numpy as np
import Bench.MComp as mcomp
import Bench.TS_datasets as tsds
import warnings

tester7 = mcomp.cMComp_Tester(tsds.load_M4_comp() , "M4_COMP");
with warnings.catch_warnings():
    warnings.simplefilter("error")
    # tester7.testSignal('ECON0137')
    # tester7.testSignal('ECON0299')
    # tester7.testAllSignals()
    tester7.run_multiprocessed(20);
