# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np
import pyaf.Bench.TS_datasets as tsds

import sys, os, traceback

import pyaf.Bench.GenericBenchmark as ben


class cArtificial_Tester(ben.cGeneric_Tester):

    '''
   
    '''
        
    def __init__(self , tsspec, bench_name):
        super().__init__(tsspec, bench_name);
        pass;

