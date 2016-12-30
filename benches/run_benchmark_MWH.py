import pandas as pd
import numpy as np

import ForecastEngine as autof
import Bench.TS_datasets as tsds

import CodeGen.TS_CodeGenerator as tscodegen
import Bench.GenericBenchmark as tBench
import warnings

#%matplotlib inline

tester1 = tBench.cGeneric_Tester(tsds.load_MWH_datsets() , "MWH_BENCH");
with warnings.catch_warnings():
    warnings.simplefilter("error")
    # tester1.testSignals('plastics')
    # tester1.testAllSignals()
    tester1.run_multiprocessed();

