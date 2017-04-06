# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np
import datetime
from . import Utils as tsutil

def testTranform(tr1):
    df = pd.DataFrame();
    df['A'] = np.random.normal(0, 1.0, 10);
    sig = df['A'];

    tr1.mOriginalSignal = "selfTestSignal";
    tr1.fit(sig)
    sig1 = tr1.apply(sig);
    sig2 = tr1.invert(sig1)
    n = np.linalg.norm(sig2 - sig)
    print("'" + tr1.get_name("Test") + "'" , " : ", n)
    if(n > 1.0e-10):
        print(sig.values)
        print(sig1.values)
        print(sig2.values)    

    assert(n <= 1.0e-10)    

class cAbstractSignalTransform:
    def __init__(self):
        self.mOriginalSignal = None;
        self.mComplexity = None;
        self.mScaling = None;
        self.mDebug = False;
        pass

    def is_applicable(self, sig):
        return True;

    def fit_scaling_params(self, sig):
        if(self.mScaling is not None):
            self.mMeanValue = np.mean(sig);
            self.mStdValue = np.std(sig);
            self.mMinValue = np.min(sig);
            self.mMaxValue = np.max(sig);
            self.mDelta = self.mMaxValue - self.mMinValue;
            eps = 1.0e-10
            if(self.mDelta < eps):
                self.mDelta = eps;
        else:
            return sig;

    def scale_signal(self, sig):
        if(self.mScaling is not None):
            # print("SCALE_START", sig.values[1:5]);
            sig1 = sig.copy();
            sig1[sig1 <= self.mMinValue] = self.mMinValue;
            sig1[sig1 >= self.mMaxValue] = self.mMaxValue;
            sig1 = (sig1 - self.mMinValue) / self.mDelta;
            # print("SCALE_END", sig1.values[1:5]);
            return sig1;
        else:
            return sig;

    def rescale_signal(self, sig1):
        if(self.mScaling is not None):
            # print("RESCALE_START", sig1.values[1:5]);
            sig = sig1.copy();
            sig[sig > 1.0] = 1.0;
            sig[sig < 0.0] = 0.0;
            sig = sig * self.mDelta + self.mMinValue;
            # print("RESCALE_END", sig.values[1:5]);
            return sig;
        else:
            return sig1;

    def fit(self , sig):
        # print("FIT_START", self.mOriginalSignal, sig.values[1:5]);
        self.fit_scaling_params(sig);
        sig1 = self.scale_signal(sig);
        self.specific_fit(sig1);
        # print("FIT_END", self.mOriginalSignal, sig1.values[1:5]);
        pass
    
    def apply(self, sig):
        # print("APPLY_START", self.mOriginalSignal, sig.values[1:5]);
        sig1 = self.scale_signal(sig);
        sig2 = self.specific_apply(sig1);
        # print("APPLY_END", self.mOriginalSignal, sig2.values[1:5]);
        if(self.mDebug):
            self.check_not_nan(sig2 , "transform_apply");
        return sig2;

    def invert(self, sig1):
        # print("INVERT_START", self.mOriginalSignal, sig1.values[1:5]);
        sig2 = self.specific_invert(sig1);
        rescaled_sig = self.rescale_signal(sig2);
        # print("INVERT_END", self.mOriginalSignal, rescaled_sig.values[1:5]);
        return rescaled_sig;


    def transformDataset(self, df, isig):
        df[self.get_name(isig)] = self.apply(df[isig])
        return df;

    def test(self):
        import copy;
        # tr1 = copy.deepcopy(self);
        # testTranform(tr1);

    def dump_apply_invert(self, df_before_apply, df_after_apply):
        df = pd.DataFrame();
        df['before_apply'] = df_before_apply;
        df['after_apply'] = df_after_apply;
        print("dump_apply_invert_head", df.head());
        print("dump_apply_invert_tail", df.tail());
        
    def check_not_nan(self, sig , name):
        if(np.isnan(sig).any()):
            print("TRANSFORMATION_RESULT_WITH_NAN_IN_SIGNAL" , sig);
            raise tsutil.InternalForecastError("Invalid transformation for column '" + name + "'");
        pass


class cSignalTransform_None(cAbstractSignalTransform):

    def __init__(self):
        cAbstractSignalTransform.__init__(self);
        self.mFormula = "None";
        self.mComplexity = 0;
        pass

    def get_name(self, iSig):
        return "_" + iSig;
    
    def specific_fit(self , sig):
        pass
    
    def specific_apply(self, df):
        return df;
    
    def specific_invert(self, df):
        return df;

        

class cSignalTransform_Accumulate(cAbstractSignalTransform):

    def __init__(self):
        cAbstractSignalTransform.__init__(self);
        self.mFormula = "Integration";
        self.mComplexity = 1;
        pass

    def get_name(self, iSig):
        return "CumSum_" + iSig;
    
    def specific_fit(self , sig):
        pass
    
    def specific_apply(self, sig):
        return sig.cumsum(axis = 0);
    
    def specific_invert(self, df):
        df_orig = df.copy();
        df_orig.iloc[0] = df.iloc[0];
        for i in range(1,df.shape[0]):
            df_orig.iloc[i] = df.iloc[i] -  df.iloc[i - 1]
        return df_orig;


class cSignalTransform_Quantize(cAbstractSignalTransform):

    def __init__(self, iQuantiles):
        cAbstractSignalTransform.__init__(self);
        self.mQuantiles = iQuantiles;
        self.mFormula = "Quantization";
        self.mComplexity = 2;
        pass

    def get_name(self, iSig):
        return "Quantized_" + str(self.mQuantiles) + "_" + iSig;

    def is_applicable(self, sig):
        N = sig.shape[0];
        if(N < (5 * self.mQuantiles)) :
            return False;
        return True;
    
    def specific_fit(self , sig):
        Q = self.mQuantiles;
        q = pd.Series(range(0,Q)).apply(lambda x : sig.quantile(x/Q))
        self.mCurve = q.to_dict()
        pass

    def signal2quant(self, x):
        curve = self.mCurve;
        return min(curve.keys(), key=lambda y:abs(float(curve[y])-x))
    
    def specific_apply(self, df):
        lSignal_Q = df.apply(lambda x : self.signal2quant(x));
        return lSignal_Q;

    def quant2signal(self, x):
         curve = self.mCurve;
         key = int(x);
         (lMin, lMax) = (min(self.mCurve.keys()), max(self.mCurve.keys()))
         if(key >= lMax):
             key = lMax;
         if(key <= lMin):
             key = lMin;            
         val = curve[key]
         return val;

    def specific_invert(self, df):
        lSignal = df.apply(lambda x : self.quant2signal(x));
        return lSignal;


class cSignalTransform_BoxCox(cAbstractSignalTransform):

    def __init__(self, iLambda):
        cAbstractSignalTransform.__init__(self);
        self.mFormula = "BoxCox";
        self.mLambda = iLambda;
        self.mComplexity = 2;
        self.mScaling = True;
        pass

    def get_name(self, iSig):
        return "Box_Cox_" + str(self.mLambda) + "_" + iSig;

    def specific_fit(self, sig):
        self.mFormula = "BoxCox(Lambda=" + str(self.mLambda) + ")";
        pass
    

    def specific_apply(self, df):
        log_df = np.log(df + 1);
        if(abs(self.mLambda) <= 0.01):
            return log_df;
        return (np.exp(log_df * self.mLambda) - 1) / self.mLambda;
    
    def specific_invert(self, df):
        df1 = df.copy();
        if(abs(self.mLambda) <= 0.01):
            df_orig = np.exp(df1) - 1;
            return df_orig;
        df1 [df1 <= (-1.0/self.mLambda)] = 1.0e-8;
        y = np.log(self.mLambda * df1 + 1) / self.mLambda;
        df_pos = np.exp(y) - 1;
        return df_pos;



class cSignalTransform_Differencing(cAbstractSignalTransform):

    def __init__(self):
        cAbstractSignalTransform.__init__(self);
        self.mFirstValue = None;
        self.mFormula = "Difference";
        self.mComplexity = 1;
        pass

    def get_name(self, iSig):
        return "Diff_" + iSig;

    def specific_fit(self, sig):
        # print(sig.head());
        self.mFirstValue = sig.iloc[0];
        pass
    

    def specific_apply(self, df):
        df_shifted = df.shift(1)
        df_shifted.iloc[0] = self.mFirstValue;
        return (df - df_shifted);
    
    def specific_invert(self, df):
        df_orig = df.copy();
        df_orig.iloc[0] = self.mFirstValue;
        for i in range(1,df.shape[0]):
            df_orig.iloc[i] = df.iloc[i] +  df_orig.iloc[i - 1]
        return df_orig;


class cSignalTransform_RelativeDifferencing(cAbstractSignalTransform):

    def __init__(self):
        cAbstractSignalTransform.__init__(self);
        self.mFirstValue = None;
        self.mFormula = "RelativeDifference";
        self.mComplexity = 1;
        self.mScaling = True;
        pass

    def get_name(self, iSig):
        return "RelDiff_" + iSig;

    def specific_fit(self, sig):
        self.mFirstValue = sig.iloc[0];
        pass

    def specific_apply(self, df):
        df_shifted = df.shift(1)
        df_shifted.iloc[0] = self.mFirstValue;
        rate = (df - df_shifted) / (df_shifted + 1.0)
        return rate;
    
    def specific_invert(self, df):
        # print("RelDiff_DEBUG" , self.mFirstValue, df.values);
        rate = df;
        df_orig = 0.0 * rate;
        df_orig.iloc[0] = self.mFirstValue;

        for i in range(1,df.shape[0]):
            df_orig.iloc[i] = (df_orig.iloc[i-1] + 1) * rate.iloc[i] + df_orig.iloc[i-1];

        return df_orig;


class cSignalTransform_Logit(cAbstractSignalTransform):

    def __init__(self):
        cAbstractSignalTransform.__init__(self);
        self.mFormula = "Logit";
        self.mComplexity = 1;
        self.mScaling = True;
        pass

    def get_name(self, iSig):
        return "Logit_" + iSig;


    def is_applicable(self, sig):
        if(self.mScaling is not None):
            return True;
        # this has to be a proportion ( 0 <= p <= 1.0 )
        lMinValue = np.min(sig);
        lMaxValue = np.max(sig);
        if((lMinValue >= 0.0) and (lMaxValue <= 1.0)):
            return True;
        return False;

    def specific_fit(self, sig):
        pass

    def logit(self, x):
        eps = 1.0e-8;
        x1 = x;
        if(x < eps):
            x1 = eps;
        if(x > (1.0 - eps)):
            x1 = 1.0 - eps;
        y = np.log(x1) - np.log(1 - x1);
        return y;

    def inv_logit(self, y):
        x = np.exp(y);
        p = x / (1 + x);
        return p;

    def specific_apply(self, df):
        # logit
        df1 = df.apply(lambda x : self.logit(x));
        return df1;
    
    def specific_invert(self, df):
        # logit
        df1 = df.apply(lambda x : self.inv_logit(x));
        return df1;


        

class cSignalTransform_Anscombe(cAbstractSignalTransform):
    '''
    More suitable for poissonnian signals (counts)
    See https://en.wikipedia.org/wiki/Anscombe_transform
    '''

    def __init__(self):
        cAbstractSignalTransform.__init__(self);
        self.mComplexity = 1;
        self.mFormula = "Anscombe";
        self.mConstant = 3.0/ 8.0;
        pass

    def get_name(self, iSig):
        return "Anscombe_" + iSig;
    
    def specific_fit(self , sig):
        pass
    
    def specific_apply(self, sig):
        y = 2 * np.sqrt(sig.values + self.mConstant);
        return y;
    
    def specific_invert(self, sig):
        x = (sig / 2) *  (sig / 2) - self.mConstant ;
        return x;


class cSignalTransform_Fisher(cAbstractSignalTransform):
    '''
    https://en.wikipedia.org/wiki/Fisher_transformation
    '''

    def __init__(self):
        cAbstractSignalTransform.__init__(self);
        self.mFormula = "Fisher";
        self.mComplexity = 1;
        self.mScaling = True;
        pass

    def get_name(self, iSig):
        return "Fisher_" + iSig;
    
    def specific_fit(self , sig):
        pass
    
    def specific_apply(self, sig):
        eps = 1.0e-8;
        y = np.arctanh(np.clip(sig.values , -1 + eps , 1.0 - eps));
        return y;
    
    def specific_invert(self, sig):
        x = np.tanh(sig.values);
        return x;
