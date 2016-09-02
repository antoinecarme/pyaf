import pandas as pd
import numpy as np
import AutoForecast as autof
import Bench.TS_datasets as tsds

import sys, os, traceback

import Bench.GenericBenchmark as ben


class cArtificial_Tester(ben.cGeneric_Tester):

    '''
   
    '''
        
    def __init__(self , tsspec, bench_name):
        super().__init__(tsspec, bench_name);
        pass;

