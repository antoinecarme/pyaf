import pandas as pd
import numpy as np

import pyaf.TS.Perf as tsperf

def test_perf(x , y):
    lPerf = tsperf.cPerf();
    lPerf.compute(x, y, "xxx");
    print(x.shape , lPerf.mL2 , lPerf.mMAPE);


for n in range(200, 1000000, 10000):
    df = pd.DataFrame();
    df['x'] = range(n);
    df['y'] = np.sin(df['x']);
    test_perf(df.x, df.y)
    
