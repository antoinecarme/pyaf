import pandas as pd
import numpy as np
import AutoForecast as autof
import Bench.TS_datasets as tsds

#get_ipython().magic('matplotlib inline')

b1 = tsds.load_ozone()
df = b1.mPastData

#df.tail(10)
#df[:-10].tail()
#df[:-10:-1]
#df.describe()


lAutoF = autof.cAutoForecast()
lAutoF

H = b1.mHorizon;

N = df.shape[0];
for n in range(2*H,  N , 10):
    df1 = df.head(n).copy();
    lAutoF.train(df1 , b1.mTimeVar , b1.mSignalVar, H);
    lAutoF.getModelInfo();
    lAutoF.mSignalDecomposition.mBestTransformation.mTimeInfo.mResolution
    lSQL = lAutoF.generateCode();
