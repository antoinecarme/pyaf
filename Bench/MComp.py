import pandas as pd
import numpy as np
import AutoForecast as autof
import Bench.TS_datasets as tsds
import sys,os

import Bench.GenericBenchmark as ben




class cMComp_Tester(ben.cGeneric_Tester):

    '''
    info : https://en.wikipedia.org/wiki/Makridakis_Competitions
    
    tester1 = cMComp_Tester(tsds.load_M1_comp());
    tester1.testAllSignals();

    tester2 = cMComp_Tester(tsds.load_M2_comp());
    tester2.testAllSignals();    
    '''
        
    def __init__(self , tsspec, bench_name):
        super().__init__(tsspec , bench_name);
        pass
