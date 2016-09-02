import pandas as pd
import numpy as np
import Bench.TS_datasets as tsds
import Bench.MComp as mcomp
import warnings


tester1 = mcomp.cMComp_Tester(tsds.load_M1_comp() , "M1_COMP");


with warnings.catch_warnings():
    warnings.simplefilter("error")
    #tester1.testSignals('QNG7')
    #tester1.testSignals('QNB16 QNB2 QNG20 QNG7 QNG8 QNI7 QNM9 QRF2 YAB8 YAC22 YAC9 YAD19 YAD6 YAF15 YAF2 YAG13 YAG1 YAG26 YAI10 YAI22 YAI35 YAI9 YAM12 YAM25')
    # tester1.testAllSignals()
    tester1.run_multiprocessed()
