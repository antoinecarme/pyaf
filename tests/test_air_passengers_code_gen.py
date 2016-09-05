import pandas as pd
import numpy as np
import Bench.TS_datasets as tsds
import AutoForecast as autof
import TS_CodeGen_Objects as tscodegen

b1 = tsds.load_airline_passengers()
df = b1.mPastData

df.head()


lAutoF = autof.cAutoForecast()
lAutoF

H = b1.mHorizon;


N = df.shape[0];
for n in range(2*H,  N , 10):
    df1 = df.head(n).copy();
    lAutoF.train(df1 , b1.mTimeVar , b1.mSignalVar, H);
    lAutoF.getModelInfo();
    lAutoF.mSignalDecomposition.mBestTransformation.mTimeInfo.mResolution
    lCodeGenerator = tscodegen.cDecompositionCodeGenObject();
    lSQL = lCodeGenerator.testGeneration(lAutoF);
