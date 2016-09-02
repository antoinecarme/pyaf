import pandas as pd
import numpy as np
import Bench.TS_datasets as tsds
import Bench.Artificial as art
import warnings

def generate_datasets():
    datasets = {};
    for N in range(1, 20):
        for trend in ["constant" , "linear" , "poly"]:
            for cycle_length in range(0, 5):
                for transf in ["" , "exp"]:            
                    for sigma in range(0, 10):
                        for seed in range(0, 2):
                            ds = tsds.generate_random_TS(N * 5 , 'D', seed, trend, cycle_length, transf, sigma * 0.1);
                            datasets[ds.mName] = ds
    return datasets;


datasets = generate_datasets();
print("ARTIFICIAL_DATASETS_TESTED" , len(datasets))
tester = art.cArtificial_Tester(datasets , "ARTIFICIAL");


with warnings.catch_warnings():
    warnings.simplefilter("error")
    tester.run_multiprocessed(18);
                        
