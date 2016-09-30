import pandas as pd
import numpy as np
import warnings

import AutoForecast.ForecastEngine as autof
import AutoForecast.Bench.TS_datasets as tsds

import AutoForecast.CodeGen.TS_CodeGenerator as tscodegen
import AutoForecast.Bench.MComp as mcomp

tester7 = mcomp.cMComp_Tester(tsds.load_M4_comp() , "M4_COMP");
with warnings.catch_warnings():
    warnings.simplefilter("error")
    tester7.testSignals('ECON0137')
    # tester7.testSignal('ECON0299')
    # tester7.testAllSignals()
    # tester7.run_multiprocessed(20);
