import pandas as pd
import numpy as np

import sys,os
# for timing
import time

import multiprocessing as mp

import warnings

def createDirIfNeeded(dirname):
    try:
        os.mkdir(dirname);
    except:
        pass

class cBenchmarkError(Exception):
    def __init__(self, msg):
        super().__init__(msg);
        pass
    
    
def run_bench_process(a):
    try:
        print("RUNNING" , a.mName);
        a.mResult = a.mValues.mean();
        print(a.mResult);
        return a;
    except cBenchmarkError as error:
        print("BENCHMARKING_ERROR '" + a.mName + "'");
        logger.error(error)
        pass;
    except:
        print("BENCHMARK_FAILURE '" + a.mName + "'");
        raise

class cGeneric_Tester_Arg:
    def __init__(self , name, values):
        self.mName = name;
        self.mValues = values;
        self.mResult = None;

def run_multiprocessed(nbprocesses = 20):
    pool = mp.Pool(nbprocesses)
    args = []
    for sig in range(100):
        values = np.arange(1000000);
        a = cGeneric_Tester_Arg("PYAF_SYSTEM_DEPENDENT_process_" + str(sig), values);
        args = args + [a];
        
    out = {};
    for res in pool.imap(run_bench_process, args):
        out[res.mName] = res;
    pool.close()
    pool.join()
    return out;

def run():
    results = run_multiprocessed(nbprocesses = 12);
    for res in results.keys():
        print(res , results[res].mResult);


with warnings.catch_warnings():
    warnings.simplefilter("error")
    run();
