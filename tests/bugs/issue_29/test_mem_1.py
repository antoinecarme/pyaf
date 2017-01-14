from __future__ import absolute_import

import pandas as pd
import numpy as np


def display_used_mem():
    print("DISPLAY_USED_MEM_START");
    import gc
    gc.collect()
    import objgraph
    objgraph.show_most_common_types(limit=20)
    print("DISPLAY_USED_MEM_END");
    

# import '.' as pyaf_new_name
# pyaf=pyaf_new_name
# from pyaf

display_used_mem();
import pyaf.Bench.TS_datasets as tsds

#get_ipython().magic('matplotlib inline')

display_used_mem();
b1 = tsds.load_ozone()
df = b1.mPastData

display_used_mem();
df.info();

display_used_mem();

