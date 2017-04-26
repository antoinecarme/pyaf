import warnings

import pyaf.Bench.TS_datasets as tsds

import pyaf.Bench.MComp as mcomp

# M4Comp_BUSINESS-INDUSTRY.csv.gz
# M4Comp_DEMOGRAPHICS.csv.gz
# M4Comp_FINANCE.csv.gz
# M4Comp_INVENTORY.csv.gz
# M4Comp_CLIMATE.csv.gz
# M4Comp_ECONOMICS.csv.gz
# M4Comp_INTERNET-TELECOM.csv.gz



tester7 = mcomp.cMComp_Tester(tsds.load_M4_comp("ECONOMICS") , "M4_COMP");
with warnings.catch_warnings():
    warnings.simplefilter("error")
    tester7.testSignals('ECON0137')
    # tester7.testSignal('ECON0299')
    # tester7.testAllSignals()
    # tester7.run_multiprocessed(20);
