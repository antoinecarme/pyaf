import pandas as pd
import numpy as np
import AutoForecast.ForecastEngine as autof
import AutoForecast.Bench.TS_datasets as tsds

import AutoForecast.CodeGen.TS_CodeGenerator as tscodegen
import warnings
import AutoForecast.Bench.MComp as mcomp


tester2 = mcomp.cMComp_Tester(tsds.load_M2_comp() , "M2_COMP");

with warnings.catch_warnings():
    warnings.simplefilter("error")
    tester1.testSignal('ADDD')
    # tester2.testAllSignals()
    # tester2.run_multiprocessed();
