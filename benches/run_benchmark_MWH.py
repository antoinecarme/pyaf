import pandas as pd
import numpy as np

import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds

import pyaf.CodeGen.TS_CodeGenerator as tscodegen
import pyaf.Bench.GenericBenchmark as tBench
import warnings

#%matplotlib inline

tester1 = tBench.cGeneric_Tester(tsds.load_MWH_datsets() , "MWH_BENCH");
with warnings.catch_warnings():
    warnings.simplefilter("error")
    # tester1.testSignals('plastics')
    # tester1.testAllSignals()
    tester1.run_multiprocessed();

