import pandas as pd
import numpy as np
import Bench.MComp as mcomp
import Bench.TS_datasets as tsds
import warnings

lTypes = ["BUSINESS-INDUSTRY", "DEMOGRAPHICS", "FINANCE", "INVENTORY", "CLIMATE",  "ECONOMICS", "INTERNET-TELECOM"]
# lTypes = ["BUSINESS-INDUSTRY"];

for ty in lTypes:
    tester = mcomp.cMComp_Tester(tsds.load_M4_comp(ty) , "M4_COMP_" + ty);
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # tester7.testSignal('ECON0137')
        # tester7.testSignal('ECON0299')
        # tester7.testAllSignals()
        tester.run_multiprocessed();
