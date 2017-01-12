from __future__ import absolute_import

import pandas as pd
import numpy as np


# import '.' as pyaf_new_name
# pyaf=pyaf_new_name
# from pyaf
import Bench.TS_datasets as tsds

#get_ipython().magic('matplotlib inline')

b1 = tsds.load_ozone()
df = b1.mPastData

df.info();
