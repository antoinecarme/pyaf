import pandas as pd
import numpy as np
import pyaf.Bench.TS_datasets as tsds
import pyaf.Bench.Artificial as art
import warnings


with warnings.catch_warnings():
    warnings.simplefilter("error")
    tester1 = art.cArtificial_Tester(tsds.load_artificial_datsets("S") , "ARTIFICIAL_S");
    tester1.run_multiprocessed(18);
    del tester1;
    tester2 = art.cArtificial_Tester(tsds.load_artificial_datsets("M") , "ARTIFICIAL_M");
    tester2.run_multiprocessed(18);
    del tester2;
    tester3 = art.cArtificial_Tester(tsds.load_artificial_datsets("L") , "ARTIFICIAL_L");
    tester3.run_multiprocessed(18);
    del tester3;
    tester4 = art.cArtificial_Tester(tsds.load_artificial_datsets("XL") , "ARTIFICIAL_XL");
    tester4.run_multiprocessed(18);
    del tester4;
    
                        
