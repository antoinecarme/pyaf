# Copyright (C) 2016 Antoine Carme <Antoine.Carme@outlook.com>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np

from . import Utils as tsutil

def testTransform_one_seed(tr1 , seed_value):
    df = pd.DataFrame(index = None);
    np.random.seed(seed_value)
    df['A'] = np.random.normal(0, 1.0, 10);
    # df['A'] = range(1, 6000);
    sig = df['A'];

    tr1.mOriginalSignal = "selfTestSignal";
    tr1.fit(sig)
    sig1 = tr1.apply(sig);
    sig2 = tr1.invert(sig1)
    # print(sig)
    # print(sig1)
    # print(sig2)
    n = np.linalg.norm(sig2 - sig)
    lEps = 1e-7
    if(n > lEps):
        print("'" + tr1.get_name("Test") + "'" , " : ", n)
        print(sig.values)
        print(sig1.values)
        print(sig2.values)    

    assert(n <= lEps)    


def testTransform(tr1):
    for seed_value in range(0,10,100):
        testTransform_one_seed(tr1, seed_value)

class cAbstractSignalTransform:
    def __init__(self):
        self.mOriginalSignal = None;
        self.mComplexity = None;
        self.mScaling = None;
        self.mDebug = False;
        pass

    def is_applicable(self, sig):
        return True;


    def checkSignalType(self, sig):
        # print(df.info());
        type2 = sig.dtype
        if(type2.kind == 'O'):
            raise tsutil.PyAF_Error('Invalid Signal Column Type ' + sig.dtype);

    def fit_scaling_params(self, sig):
        if(self.mScaling is not None):
            # self.mMeanValue = np.mean(sig);
            # self.mStdValue = np.std(sig);
            # lEps = 1.0e-10
            self.mMinInputValue = np.min(sig);
            self.mMaxInputValue = np.max(sig);
            self.mInputValueRange = self.mMaxInputValue - self.mMinInputValue;
        else:
            pass

    def scale_value(self, x):
        return (x - self.mMinInputValue) / self.mInputValueRange;

    def scale_signal(self, sig):
        if(self.mScaling is not None):
            # print("SCALE_START", sig.values[1:5]);
            sig1 = None
            if(np.fabs(self.mInputValueRange) < 1e-10):
                sig1 = sig * 0.0
            else:
                sig1 = self.scale_value(sig)
            # print("SCALE_END", sig1.values[1:5]);
            return sig1;
        else:
            return sig;

    def rescale_value(self, x):
        y = self.mMinInputValue + x * self.mInputValueRange;
        return y
    
    
        

    def rescale_signal(self, sig1):
        if(self.mScaling is not None):
            # print("RESCALE_START", sig1.values[1:5]);
            sig = self.rescale_value(sig1);
            # print("RESCALE_END", sig.values[1:5]);
            return sig;
        else:
            return sig1;

    def fit(self , sig):
        # print("FIT_START", self.mOriginalSignal, sig.values[1:5]);
        self.checkSignalType(sig)
        self.fit_scaling_params(sig);
        sig1 = self.scale_signal(sig);
        self.specific_fit(sig1);
        # print("FIT_END", self.mOriginalSignal, sig1.values[1:5]);
        pass

    def apply(self, sig):
        # print("APPLY_START", self.mOriginalSignal, sig.values[1:5]);
        self.checkSignalType(sig)
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
        # import copy;
        # tr1 = copy.deepcopy(self);
        # testTransform(tr1);
        pass

    def dump_apply_invert(self, sig_before_apply, sig_after_apply):
        sig = pd.Series(index = None);
        sig['before_apply'] = sig_before_apply;
        sig['after_apply'] = sig_after_apply;
        print("dump_apply_invert_head", sig.head());
        print("dump_apply_invert_tail", sig.tail());
        
    def check_not_nan(self, sig , name):
        if(np.isnan(sig).any()):
            print("TRANSFORMATION_RESULT_WITH_NAN_IN_SIGNAL" , sig);
            raise tsutil.Internal_PyAF_Error("Invalid transformation for column '" + name + "'");
        pass


class cSignalTransform_None(cAbstractSignalTransform):

    def __init__(self):
        cAbstractSignalTransform.__init__(self);
        self.mFormula = "NoTransf";
        self.mComplexity = 0;
        self.mScaling = True;
        pass

    def get_name(self, iSig):
        return "_" + str(iSig);
    
    def specific_fit(self , sig):
        pass
    
    def specific_apply(self, sig):
        return sig;
    
    def specific_invert(self, sig):
        return sig;

    def dump_values(self):
        logger = tsutil.get_pyaf_logger();
        logger.info("SIGNAL_TRANSFORMATION_MODEL_VALUES " + self.mFormula + " " + str(None));
        

class cSignalTransform_Accumulate(cAbstractSignalTransform):

    def __init__(self):
        cAbstractSignalTransform.__init__(self);
        self.mFormula = "Integration";
        self.mComplexity = 1;
        self.mScaling = True;
        pass

    def get_name(self, iSig):
        return "CumSum_" + str(iSig);
    
    def specific_fit(self , sig):
        pass
    
    def specific_apply(self, sig):
        return sig.cumsum(axis = 0)
    
    def specific_invert(self, sig):
        sig_orig = sig - sig.shift(1);
        sig_orig.iloc[0] = sig.iloc[0];
        return sig_orig;

    def dump_values(self):
        logger = tsutil.get_pyaf_logger();
        logger.info("SIGNAL_TRANSFORMATION_MODEL_VALUES " + self.mFormula + " " + str(None));

class cSignalTransform_Quantize(cAbstractSignalTransform):

    def __init__(self, iQuantiles):
        cAbstractSignalTransform.__init__(self);
        self.mQuantiles = iQuantiles;
        self.mFormula = "Quantization";
        self.mComplexity = 2;
        self.mScaling = True;
        pass

    def get_name(self, iSig):
        return "Quantized_" + str(self.mQuantiles) + "_" + str(iSig);

    def is_applicable(self, sig):
        N = sig.shape[0];
        if(N < (5 * self.mQuantiles)) :
            return False;
        return True;
    
    def specific_fit(self , sig):
        Q = self.mQuantiles;
        q = pd.Series(range(0,Q)).apply(lambda x : sig.quantile(x/Q))
        self.mCurve = q.to_dict()
        (self.mMin, self.mMax) = (min(self.mCurve.keys()), max(self.mCurve.keys()))
        pass

    def signal2quant(self, x):
        curve = self.mCurve;
        return min(curve.keys(), key=lambda y:abs(float(curve[y])-x))
    
    def specific_apply(self, sig):
        lSignal_Q = sig.apply(self.signal2quant);
        return lSignal_Q;

    def quant2signal(self, x):
         curve = self.mCurve;
         key = int(x);
         if(key >= self.mMax):
             key = self.mMax;
         if(key <= self.mMin):
             key = self.mMin;            
         val = curve[key]
         return val;

    def specific_invert(self, sig):
        lSignal = sig.apply(self.quant2signal);
        return lSignal;

    def dump_values(self):
        logger = tsutil.get_pyaf_logger();
        logger.info("QUANTIZE_TRANSFORMATION_MIN_MAX_CURVE " + self.mFormula + " " + str((self.mMin , self.mMax)) + " " + str(self.mCurve));



class cSignalTransform_BoxCox(cAbstractSignalTransform):

    def __init__(self, iLambda):
        cAbstractSignalTransform.__init__(self);
        self.mFormula = "BoxCox";
        self.mLambda = iLambda;
        self.mComplexity = 2 + abs(self.mLambda);
        self.mScaling = True;
        pass

    def get_name(self, iSig):
        return "Box_Cox_" + str(self.mLambda) + "_" + str(iSig);

    def specific_fit(self, sig):
        self.mFormula = "BoxCox(Lambda=" + str(self.mLambda) + ")";
        pass
    

    def specific_apply(self, sig):
        lEps = 1e-3
        assert(sig.min() > -lEps)
        log_sig = np.log(sig.clip(lEps, None))
        if(abs(self.mLambda) <= 0.001):
            return log_sig;
        lLimit = 5.0 / abs(self.mLambda)
        log_sig = log_sig.clip(-lLimit , lLimit)
        sig1 = (np.exp(log_sig * self.mLambda) - 1) / self.mLambda
        return sig1;

    def invert_value(self, y):
        x = y;
        lEps = 1e-5
        x0 = self.mLambda * x + 1
        x1 = np.log(x0.clip(lEps, None)) / self.mLambda;
        return np.exp(x1).clip(0, 1) ;        
    
    def specific_invert(self, sig):
        if(abs(self.mLambda) <= 0.001):
            sig_orig = np.exp(sig).clip(0, 1);
            return sig_orig;
        sig_pos = self.invert_value(sig)
        return sig_pos;

    def dump_values(self):
        logger = tsutil.get_pyaf_logger();
        logger.info("BOX_COX_TRANSFORMATION_LAMBDA " + self.mFormula + " " + str(self.mLambda));


class cSignalTransform_Differencing(cAbstractSignalTransform):

    def __init__(self):
        cAbstractSignalTransform.__init__(self);
        self.mFirstValue = None;
        self.mFormula = "Difference";
        self.mComplexity = 1;
        self.mScaling = True;
        pass

    def get_name(self, iSig):
        return "Diff_" + str(iSig);

    def specific_fit(self, sig):
        # print(sig.head());
        self.mFirstValue = sig.iloc[0];
        pass
    

    def specific_apply(self, sig):
        sig_shifted = sig.shift(1)
        sig_shifted.iloc[0] = self.mFirstValue;
        lResult = sig - sig_shifted
        return lResult
    
    def specific_invert(self, sig):
        sig_cumsum = sig.cumsum();
        sig_orig = sig_cumsum + self.mFirstValue;
        return sig_orig;

    def dump_values(self):
        logger = tsutil.get_pyaf_logger();
        logger.info("DIFFERENCING_TRANSFORMATION " + self.mFormula + " " + str(self.mFirstValue));


class cSignalTransform_RelativeDifferencing(cAbstractSignalTransform):

    def __init__(self):
        cAbstractSignalTransform.__init__(self);
        self.mFirstValue = None;
        self.mFormula = "RelativeDifference";
        self.mComplexity = 1;
        self.mScaling = True;
        pass

    def get_name(self, iSig):
        return "RelDiff_" + str(iSig);
    
    def specific_fit(self, sig):
        self.mFirstValue = sig.iloc[0];
        pass

    def specific_apply(self, sig):
        lEps = 1e-2
        # print("RelDiff_apply_DEBUG_START" , self.mFirstValue, sig.values[0:10]);
        sig1 = sig
        sig1[sig1.abs() <= lEps] = lEps
        sig_shifted = sig1.shift(1)
        # sig_shifted[sig_shifted <= lEps] = lEps
        rate = (sig1 - sig_shifted) / sig_shifted
        rate.iloc[0] = 0.0;
        # print(sig1)
        # print(sig_shifted)
        rate = rate.clip(-1.0e+2 , +1.0e+2)
        # print("RelDiff_apply_DEBUG_END" , rate[0:10]);
        return rate;


    def cumprod_no_overflow(self, rate):
        lEps = 1e-2
        lLogRate = np.log(rate.clip(lEps, +1.0e+2))
        lCumSum = lLogRate.cumsum()
        lCumSum = lCumSum.clip(lEps , +1.0e+2)
        lResult = np.exp(lCumSum)
        return lResult
        
    def specific_invert(self, sig):
        # print("RelDiff_invert_DEBUG_START" , self.mFirstValue, sig.values[0:10]);
        rate = sig + 1;
        rate = rate.clip(-1.0e+8 , +1.0e+8)
        rate_cum = self.cumprod_no_overflow(rate);
        sig_orig = rate_cum.clip(-1.0e+8 , +1.0e+8)
        sig_orig = self.mFirstValue * sig_orig;
        # print("rate" , rate)
        # print("rate_cum", rate_cum)
        # print("RelDiff_invert_DEBUG_START" , sig_orig[0:10])
        return sig_orig;

    def dump_values(self):
        logger = tsutil.get_pyaf_logger();
        logger.info("REALTIVE_DIFFERENCING_TRANSFORMATION " + self.mFormula + " " + str(self.mFirstValue));

class cSignalTransform_Logit(cAbstractSignalTransform):

    def __init__(self):
        cAbstractSignalTransform.__init__(self);
        self.mFormula = "Logit";
        self.mComplexity = 1;
        self.mScaling = True;
        pass

    def get_name(self, iSig):
        return "Logit_" + str(iSig);


    def is_applicable(self, sig):
        return True;

    def specific_fit(self, sig):
        pass

    def logit(self, x):
        eps = 1.0e-2;
        x1 = np.clip(x, eps, 1 - eps)
        y = np.log(x1) - np.log(1 - x1);
        return y;

    def inv_logit(self, y):
        y1 = np.clip(y, -5, 5)
        x = np.exp(y1);
        p = x / (1 + x);
        return p;

    def specific_apply(self, sig):
        # logit
        sig1 = self.logit(sig)
        return sig1;
    
    def specific_invert(self, sig):
        # logit
        sig1 = self.inv_logit(sig)
        return sig1;

    def dump_values(self):
        logger = tsutil.get_pyaf_logger();
        logger.info("LOGIT_TRANSFORMATION " + self.mFormula );

        

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
        self.mScaling = True;
        pass

    def get_name(self, iSig):
        return "Anscombe_" + str(iSig);
    
    def specific_fit(self , sig):
        pass
    
    def specific_apply(self, sig):
        y = 2 * np.sqrt(sig + self.mConstant)
        return y;
    
    def specific_invert(self, sig):
        y1 = sig.clip(1.22, 2.34)
        x = (y1/2 * y1/2) - self.mConstant
        return x;

    def dump_values(self):
        logger = tsutil.get_pyaf_logger();
        logger.info("ANSCOMBE_TRANSFORMATION " + self.mFormula + " " + str(self.mConstant));


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
        return "Fisher_" + str(iSig);
    
    def specific_fit(self , sig):
        pass
    
    def specific_apply(self, sig):
        eps = 1.0e-2;
        y = np.arctanh(np.clip(sig.values , -1 + eps , 1.0 - eps))
        return y;
    
    def specific_invert(self, sig):
        x = np.tanh(sig.values);
        return x;

    def dump_values(self):
        logger = tsutil.get_pyaf_logger();
        logger.info("FISCHER_TRANSFORMATION " + self.mFormula);


def create_tranformation(iName , arg):
    if(iName == 'None'):
        return cSignalTransform_None();

    if(iName == 'Difference'):
        return cSignalTransform_Differencing()
    
    if(iName == 'RelativeDifference'):
        return cSignalTransform_RelativeDifferencing()
            
    if(iName == 'Integration'):
        return cSignalTransform_Accumulate()
        
    if(iName == 'BoxCox'):
        return cSignalTransform_BoxCox(arg)
    
    if(iName == 'Quantization'):
        return cSignalTransform_Quantize(arg)
        
    if(iName == 'Logit'):
        return cSignalTransform_Logit()
        
    if(iName == 'Fisher'):
        return cSignalTransform_Fisher()
        
    if(iName == 'Anscombe'):
        return cSignalTransform_Anscombe()

    # assert(0)
    return None


class cTransformationEstimator:
    
    def __init__(self):
        self.mSignalFrame = None
        self.mTransformList = {}

    def validateTransformation(self , transf , df, iTime, iSignal):
        lName = transf.get_name("");
        lIsApplicable = transf.is_applicable(df[iSignal]);
        if(lIsApplicable):
            # print("Adding Transformation " , lName);
            self.mTransformList = self.mTransformList + [transf];


    
    def defineTransformations(self , df, iTime, iSignal):
        self.mTransformList = [];
        if(self.mOptions.mActiveTransformations['None']):
            self.validateTransformation(cSignalTransform_None() , df, iTime, iSignal);

        if(self.mOptions.mActiveTransformations['Difference']):
            self.validateTransformation(cSignalTransform_Differencing() , df, iTime, iSignal);

        if(self.mOptions.mActiveTransformations['RelativeDifference']):
            self.validateTransformation(cSignalTransform_RelativeDifferencing() , df, iTime, iSignal);
            
        if(self.mOptions.mActiveTransformations['Integration']):
            self.validateTransformation(cSignalTransform_Accumulate() , df, iTime, iSignal);

        if(self.mOptions.mActiveTransformations['BoxCox']):
            for i in self.mOptions.mBoxCoxOrders:
                self.validateTransformation(cSignalTransform_BoxCox(i) , df, iTime, iSignal);

        if(self.mOptions.mActiveTransformations['Quantization']):
            for q in self.mOptions.mQuantiles:
                self.validateTransformation(cSignalTransform_Quantize(q) , df, iTime, iSignal);
        
        if(self.mOptions.mActiveTransformations['Logit']):
            self.validateTransformation(cSignalTransform_Logit() , df, iTime, iSignal);
        
        if(self.mOptions.mActiveTransformations['Fisher']):
            self.validateTransformation(cSignalTransform_Fisher() , df, iTime, iSignal);
        
        if(self.mOptions.mActiveTransformations['Anscombe']):
            self.validateTransformation(cSignalTransform_Anscombe() , df, iTime, iSignal);
        
