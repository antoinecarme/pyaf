import pandas as pd
import numpy as np
import AutoForecast as autof
import Bench.TS_datasets as tsds
import sys,os
import Bench.GenericBenchmark as ben


class cNN_Tester(ben.cGeneric_Tester):

    '''
    tester1 = cNN_Tester(tsds.load_NN_part1());
    tester1.testAllSignals();

    tester2 = cNN_Tester(tsds.load_NN_part2());
    tester2.testAllSignals();    
    '''
        
    def __init__(self , tsspec, bench_name):
        super().__init__(tsspec, bench_name);
        pass;
