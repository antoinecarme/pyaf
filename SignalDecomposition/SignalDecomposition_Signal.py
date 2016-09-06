import pandas as pd
import numpy as np
import datetime

def testTranform(tr1):
    df = pd.DataFrame();
    df['A'] = np.random.normal(0, 1.0, 10);
    sig = df['A'];
    
    tr1.fit(sig)
    sig1 = tr1.apply(sig);
    sig2 = tr1.invert(sig1)
    n = np.linalg.norm(sig2 - sig)
#    print("'" + tr1.get_name("Test") + "'" , " : ", n)
    if(n > 1.0e-10):
        print(sig.values)
        print(sig1.values)
        print(sig2.values)    

    assert(n <= 1.0e-10)    

class cAbstractSignalTransform:
    def __init__(self):
        pass

    def dump_apply_invert(self, df_before_apply, df_after_apply):
        df = pd.DataFrame();
        df['before_apply'] = df_before_apply;
        df['after_apply'] = df_after_apply;
        print("dump_apply_invert_head", df.head());
        print("dump_apply_invert_tail", df.tail());
        


class cSignalTransform_None(cAbstractSignalTransform):

    def __init__(self):
        pass

    def get_name(self, iSig):
        return iSig;
    
    def fit(self , sig):
        pass
    
    def apply(self, df):
        return df;
    
    def invert(self, df):
        return df;

    def transformDataset(self, df, isig):
        df[self.get_name(isig)] = self.apply(df[isig])
        return df;

    def test(self):
        tr1 = cSignalTransform_None();
        testTranform(tr1);
        

class cSignalTransform_Accumulate(cAbstractSignalTransform):

    def __init__(self):
        pass

    def get_name(self, iSig):
        return "CumSum_" + iSig;
    
    def fit(self , sig):
        pass
    
    def apply(self, df):
        return df.cumsum(axis = 0);
    
    def invert(self, df):
        df_orig = df.copy();
        df_orig.iloc[0] = df.iloc[0];
        for i in range(1,df.shape[0]):
            df_orig.iloc[i] = df.iloc[i] -  df.iloc[i - 1]
        return df_orig;

    def transformDataset(self, df, isig):
        df[self.get_name(isig)] = self.apply(df[isig])
        return df;

    def test(self):
        tr1 = cSignalTransform_Accumulate();
        testTranform(tr1);
        


class cSignalTransform_Quantize(cAbstractSignalTransform):

    def __init__(self, iQuantiles):
        self.mQuantiles = iQuantiles;
        pass

    def get_name(self, iSig):
        return "Quantized_" + str(self.mQuantiles) + "_" + iSig;
    
    def fit(self , sig):
        Q = self.mQuantiles;
        q = pd.Series(range(0,Q)).apply(lambda x : sig.quantile(x/Q))
        self.mCurve = q.to_dict()
        pass

    def signal2quant(self, x):
        curve = self.mCurve;
        return min(curve.keys(), key=lambda y:abs(float(curve[y])-x))
    
    def apply(self, df):
        lSignal_Q = df.apply(lambda x : self.signal2quant(x));
        return lSignal_Q;

    def transformDataset(self, df, isig):
        df[self.get_name(isig)] = self.apply(df[isig])
        return df;

    def quant2signal(self, x):
        curve = self.mCurve;
        val = curve[int(x)]
        return val;

    def invert(self, df):
        lSignal = df.apply(lambda x : self.quant2signal(x));
        return lSignal;

    def test(self):
        tr1 = cSignalTransform_Quantize(self.mQuantiles);
        #testTranform(tr1);
        

class cSignalTransform_BoxCox(cAbstractSignalTransform):

    def __init__(self, iLambda):
        self.mLambda = iLambda;
        self.mMin = 0;
        pass

    def get_name(self, iSig):
        return "Box_Cox_" + str(self.mLambda) + "_" + iSig;

    def fit(self, sig):
        self.mMin = np.min(sig);
        self.mMax = np.max(sig);
#        print("minimum  : " , self.mMin);
        pass
    

    def apply(self, df):
        df1 = df.copy();
        df1[df1 <= self.mMin] = self.mMin;
        df1[df1 >= self.mMax] = self.mMax;
        lDelta = self.mMax - self.mMin
        df_pos = (df1 - self.mMin) / lDelta + 1
        log_df = np.log(df_pos);
        if(abs(self.mLambda) <= 0.01):
            return log_df;
        return (np.exp(log_df * self.mLambda) - 1) / self.mLambda;
    
    def invert(self, df):
        lDelta = self.mMax - self.mMin
        if(abs(self.mLambda) <= 0.01):
            df_orig = (np.exp(df) - 1 ) * lDelta + self.mMin
            return df_orig;
        y = np.log(self.mLambda * df + 1) / self.mLambda;
        df_pos = np.exp(y);
        return (df_pos - 1 ) * lDelta + self.mMin;

    def transformDataset(self, df, isig):
        df[self.get_name(isig)] = self.apply(df[isig])
        return df;


    def test(self):
        tr1 = cSignalTransform_BoxCox(self.mLambda);
        testTranform(tr1);


class cSignalTransform_Differencing(cAbstractSignalTransform):

    def __init__(self):
        self.mFirstValue = np.nan;
        pass

    def get_name(self, iSig):
        return "Diff_" + iSig;

    def fit(self, sig):
        self.mFirstValue = sig.iloc[0];
        pass
    

    def apply(self, df):
        df_shifted = df.shift(1)
        df_shifted.fillna(self.mFirstValue, inplace = True);
        return (df - df_shifted);
    
    def invert(self, df):
        df_orig = df.copy();
        df_orig.iloc[0] = self.mFirstValue;
        for i in range(1,df.shape[0]):
            df_orig.iloc[i] = df.iloc[i] +  df_orig.iloc[i - 1]
        return df_orig;

    def transformDataset(self, df, isig):
        df[self.get_name(isig)] = self.apply(df[isig])
        return df;


    def test(self):
        tr1 = cSignalTransform_Differencing();
        testTranform(tr1);


class cSignalTransform_RelativeDifferencing(cAbstractSignalTransform):

    def __init__(self):
        self.mFirstValue = np.nan;
        self.mMinValue = np.nan;
        pass

    def get_name(self, iSig):
        return "RelDiff_" + iSig;

    def fit(self, sig):
        self.mMinValue = np.min(sig);
        self.mMaxValue = np.max(sig);
        self.mFirstValue = sig.iloc[0];
        self.mDelta = self.mMaxValue - self.mMinValue;
        eps = 1.0e-10
        if(self.mDelta < eps):
            self.mDelta = eps;            
        pass

    def apply(self, df):
        df1 = df.copy();
        df1[df1 <= self.mMinValue] = self.mMinValue;
        df1[df1 >= self.mMaxValue] = self.mMaxValue;
        df1 = (df1 - self.mMinValue) / self.mDelta;
        df_shifted = df1.shift(1)
        df_shifted.fillna((self.mFirstValue - self.mMinValue) / self.mDelta, inplace = True);
        r = (df1 - df_shifted) / (df_shifted + 1)
        # self.dump_apply_invert(df , r);
        return r;
    
    def invert(self, df):
        r = df;
        df_orig = 0.0 * df;
        df_orig.iloc[0] = (self.mFirstValue - self.mMinValue) / self.mDelta
        for i in range(1,df.shape[0]):
            previous_value = df_orig.iloc[i - 1] 
            df_orig.iloc[i] = (r.iloc[i] + 1) * (previous_value + 1) - 1
        for i in range(0,df.shape[0]):
            df_orig.iloc[i] = df_orig.iloc[i] * self.mDelta + self.mMinValue;
        
        # self.dump_apply_invert(df_orig , r);
        return df_orig;

    def transformDataset(self, df, isig):
        df[self.get_name(isig)] = self.apply(df[isig])
        return df;


    def test(self):
        tr1 = cSignalTransform_RelativeDifferencing();
        testTranform(tr1);


