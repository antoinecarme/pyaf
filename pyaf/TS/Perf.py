# Copyright (C) 2016 Antoine Carme <Antoine.Carme@outlook.com>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np

from . import Utils as tsutil

class cPerf:

    def higher_values_are_better(criterion):
        return criterion in ["R2", "PEARSONR", "KendallTau", "KS", "MWU" , "AUC"]

    def __init__(self):
        self.mErrorStdDev = None;
        self.mErrorMean = None;
        self.mMAPE = None;
        self.mSMAPE = None;
        self.mDiffSMAPE = None;
        self.mMASE = None;
        self.mRMSSE = None;
        self.mL1 = None;
        self.mL2 = None;
        self.mR2 = None;
        self.mPearsonR = None;
        self.mMedAE = None;
        self.mCount = None;
        self.mName = "No_Name";
        self.mSignalQuantiles = None
        self.mCRPS = None # Continuous Ranked Probability Score ( issue #47 )
        self.mLnQ = None
        self.mKendallTau = None
        self.mMWU = None
        self.mKS = None
        self.mAUC = None
        self.mCachedValues = {}
        self.mDebug = False;

    def to_dict_summary(self, criterion):
        lDict = {"MAPE" : self.mMAPE,
                 "RMSE" : self.mL2,
                 "MAE" : self.mL1,
                 'MASE' : self.mMASE
                 }
        lCriterionValue = self.getCriterionValue(criterion)
        lDict[criterion] = lCriterionValue
        return lDict
    
    def to_dict(self):
        lDict = {"Signal" : self.mName , "Length" : self.mCount, "MAPE" : self.mMAPE,
                 "RMSE" : self.mL2,  "MAE" : self.mL1,
                 "SMAPE" : self.mSMAPE, "DiffSMAPE" : self.mDiffSMAPE,
                 'MASE' : self.mMASE, 'RMSSE' : self.mRMSSE,
                 "ErrorMean" : self.mErrorMean, "ErrorStdDev" : self.mErrorStdDev, 
                 "R2" : self.mR2, "Pearson" : self.mPearsonR, "MedAE": self.mMedAE, "LnQ" : self.mLnQ,
                 "KS" : self.mKS, "KendallTau" : self.mKendallTau, "MannWhitneyU" : self.mMWU, "AUC" : self.mAUC}
        return lDict
    
    def check_not_nan(self, sig , name):
        #print("check_not_nan");
        if(np.isnan(sig).any()):
            logger = tsutil.get_pyaf_logger();
            logger.error("PERF_WITH_NAN_IN_SIGNAL" + str(sig));
            raise tsutil.Internal_PyAF_Error("INVALID_COLUMN _FOR_PERF ['" + self.mName + "'] '" + name + "'");
        pass

    def pre_compute_abs_error_if_needed(self, signal , estimator):
        cached_result = self.mCachedValues.get('abs_error')
        if(cached_result is not None):
            return cached_result
        abs_error = np.abs(estimator.values - signal.values);
        self.mCachedValues['abs_error'] = abs_error
        return abs_error

    def pre_compute_naive_abs_error_if_needed(self, signal , estimator):
        cached_result = self.mCachedValues.get('naive_error')
        if(cached_result is not None):
            return cached_result
        naive_abs_error = np.abs(np.diff(signal.values))
        self.mCachedValues['naive_error'] = naive_abs_error
        return naive_abs_error

    def pre_compute_naive_mean_abs_error_ratio_if_needed(self, signal , estimator):
        # Used for scaled errors : MASE and RMSSE
        cached_result = self.mCachedValues.get('naive_scaled_error')
        if(cached_result is not None):
            return cached_result
        abs_error = self.pre_compute_abs_error_if_needed(signal , estimator)
        naive_abs_error= self.pre_compute_naive_abs_error_if_needed(signal , estimator)
        lEps = 1.0e-10;
        q1, q2 = 0.0, 0.0
        if(naive_abs_error.shape[0] > 0):
            naive_mean_abs_error = np.mean(naive_abs_error) + lEps
            naive_mean_abs_error_2 = np.mean(naive_abs_error * naive_abs_error) + lEps
            q1 = abs_error / naive_mean_abs_error
            q2 = abs_error * abs_error / naive_mean_abs_error_2
        self.mCachedValues['naive_scaled_error'] = (q1, q2)
        return (q1, q2)

    def compute_MAPE(self, signal , estimator):
        lEps = 1.0e-10;
        abs_error = self.pre_compute_abs_error_if_needed(signal , estimator);
        abs_rel_error = abs_error / (np.abs(signal) + lEps)
        self.mMAPE = np.mean(abs_rel_error)
        self.mMAPE = round( self.mMAPE , 4 )
        return self.mMAPE
            
    def compute_SMAPE(self, signal , estimator):
        lEps = 1.0e-10;
        abs_error = self.pre_compute_abs_error_if_needed(signal , estimator);
        sum_abs = np.abs(signal.values) + np.abs(estimator.values) + lEps
        self.mSMAPE = np.mean(2.0 * abs_error / sum_abs)
        self.mSMAPE = round( self.mSMAPE , 4 )
        return self.mSMAPE
            
    def compute_MASE(self, signal , estimator):
        (q1, q2) = self.pre_compute_naive_mean_abs_error_ratio_if_needed(signal , estimator)
        if(q1 is not None):
            self.mMASE = np.mean(q1)
            self.mMASE = round( self.mMASE , 4 )
            return self.mMASE
                
    def compute_RMSSE(self, signal , estimator):
        (q1, q2) = self.pre_compute_naive_mean_abs_error_ratio_if_needed(signal , estimator)
        if(q2 is not None):
            self.mRMSSE = np.sqrt(np.mean(q2))
            self.mRMSSE = round( self.mRMSSE , 4 )
            return self.mRMSSE
                
    def compute_DiffSMAPE(self, signal , estimator):
        abs_error = self.pre_compute_abs_error_if_needed(signal , estimator);
        lEps2 = 0.1 # for DiffSMAPE
        max_sum_eps = np.maximum(np.abs(signal.values) + np.abs(estimator.values) + lEps2,  0.5 + lEps2)
        self.mDiffSMAPE = np.mean(2.0 * abs_error / max_sum_eps)
        self.mDiffSMAPE = round( self.mDiffSMAPE , 4 )
        return self.mDiffSMAPE

    def compute_MAPE_SMAPE(self, signal, estimator):
        self.compute_MAPE(signal, estimator);
        self.compute_SMAPE(signal, estimator);
        self.compute_DiffSMAPE(signal, estimator);

    def compute_MASE_RMSSE(self, signal, estimator):
        self.compute_MASE(signal, estimator);
        self.compute_RMSSE(signal, estimator);

    def compute_R2(self, signal , estimator):
        SST = np.sum((signal.values - np.mean(signal.values))**2) + 1.0e-10;
        SSRes = np.sum((signal.values - estimator.values)**2)
        self.mR2 = 1 - SSRes/SST
        self.mR2 = round(self.mR2, 4)
        return self.mR2

    def compute_LnQ(self, signal , estimator):
        min_signal , min_estimator = signal.min() , estimator.min()
        # return +inf if the signals are not strictly positive (discard the model)
        self.mLnQ = np.Inf
        if(min_signal > 0.0 and min_estimator > 0.0):
            log_diff = np.log(estimator) - np.log(signal)
            self.mLnQ = np.sum(log_diff * log_diff)
        self.mLnQ = round( self.mLnQ , 4 )
        return self.mLnQ

    def compute_KS(self, signal , estimator):
        min_signal , max_signal = signal.min(), signal.max()
        self.mKS = 0.0
        if((max_signal - min_signal) < 0.001):
            return 
        from scipy.stats import kstest
        lKSTest = kstest(signal, estimator, method='asymp')
        self.mKS = lKSTest.statistic
        del lKSTest
        self.mKS = round( self.mKS , 4 )        
        return self.mKS
    
    def compute_Kendall(self, signal , estimator):
        min_signal , max_signal = signal.min(), signal.max()
        self.mKendallTau = 0.0
        if((max_signal - min_signal) < 0.001):
            return 
        from scipy.stats import kendalltau
        lKendallTau_Result = kendalltau(signal, estimator)
        self.mKendallTau = lKendallTau_Result.correlation
        del lKendallTau_Result
        self.mKendallTau = round( self.mKendallTau , 4 )
        return self.mKendallTau

    def pre_compute_MWU_if_needed(self, signal , estimator):
        cached_result = self.mCachedValues.get('MWU')
        if(cached_result is not None):
            return cached_result
        min_signal , max_signal = signal.min(), signal.max()
        lMWU = 0.0
        if((max_signal - min_signal) > 0.001):
            from scipy.stats import mannwhitneyu
            lMWU_Result = mannwhitneyu(signal, estimator)
            lMWU = lMWU_Result.statistic
            del lMWU_Result
            lMWU = round( lMWU, 4 )
        self.mCachedValues['MWU'] = lMWU
        return lMWU
        
    def compute_MWU(self, signal , estimator):
        self.mMWU = self.pre_compute_MWU_if_needed(signal, estimator)
        return self.mAUC
        
    def compute_AUC(self, signal , estimator):
        lMWU = self.pre_compute_MWU_if_needed(signal, estimator)
        self.mAUC = lMWU / signal.shape[0] / estimator.shape[0]
        self.mAUC = round( self.mAUC , 4 )
        return self.mAUC
        
    def dump_perf_data(self, signal , estimator):
        logger = tsutil.get_pyaf_logger();
        df = pd.DataFrame(index = signal.index);
        df['sig'] = signal.values;
        df['est'] = estimator.values;
        logger.debug(str(df.head()));
        logger.debug(str(df.tail()));
    
    def compute(self, signal , estimator, name):
        try:
            # self.dump_perf_data(signal, estimator);
            return self.real_compute(signal, estimator, name);
        except Exception as exc:
            self.dump_perf_data(signal, estimator);
            logger = tsutil.get_pyaf_logger();
            logger.error("Failure when computing perf ['" + self.mName + "'] '" + name + "'" + str(exc));
            raise tsutil.Internal_PyAF_Error("Failure when computing perf ['" + self.mName + "'] '" + name + "'");
        pass


    def compute_pearson_r(self, signal , estimator):
        from scipy.stats import pearsonr
        signal_std = np.std(signal);
        estimator_std = np.std(estimator);
        # print("PEARSONR_DETAIL2" , signal)
        # print("PEARSONR_DETAIL3" , estimator)
        lEps = 1e-4
        r = 0.0;
        if(signal_std < lEps):
            return r
        if(estimator_std < lEps):
            return r
        (r , pval) = pearsonr(signal , estimator)
        #  print("PEARSONR_DETAIL1" , signal_std, estimator_std, r)
        self.mPearsonR = round(r, 4)
        return self.mPearsonR;

    def compute_ErrorMean_ErrorStd(self, signal , estimator):
        myerror = estimator - signal 
        self.mErrorMean = np.mean(myerror)
        self.mErrorMean = round(self.mErrorMean, 4)
        self.mErrorStdDev = np.std(myerror)        
        self.mErrorStdDev = round(self.mErrorStdDev, 4)
                    
    def real_compute(self, signal , estimator, name):
        self.mCachedValues = {}
        self.mName = name;
        assert(signal.shape[0] > 0)
        if(self.mDebug):
            self.check_not_nan(signal.values , "signal")
            self.check_not_nan(estimator.values , "estimator")

        self.mCount = signal.shape[0];
        self.compute_ErrorMean_ErrorStd(signal, estimator)

        self.compute_L1(signal, estimator)
        self.compute_L2(signal, estimator)
        self.compute_MedAE(signal, estimator)        
        self.compute_MAPE_SMAPE(signal, estimator);
        self.compute_MASE_RMSSE(signal, estimator);

        
        self.mR2 = self.compute_R2(signal, estimator)
        self.mLnQ = self.compute_LnQ(signal, estimator)
        
        self.mPearsonR = self.compute_pearson_r(signal , estimator);
        self.mCRPS = self.compute_CRPS(signal , estimator);

        self.compute_KS(signal, estimator);
        self.compute_Kendall(signal, estimator);
        self.compute_MWU(signal, estimator);
        self.compute_AUC(signal, estimator);

        self.mCachedValues = {}


    def compute_signal_quantiles(self, signal , estimator):
        myerror = (estimator.values - signal.values);
        NQ = int(min(20, np.sqrt(signal.shape[0]))) # optimal quantiles number heuristics : sqrt(N)
        Q = int(100 // NQ)
        lPercentiles = [50 - q for q in range(Q, 50, Q)] + [50] + [50 + q for q in range(Q, 50, Q)]
        lPercentiles = sorted(lPercentiles)
        # print(lPercentiles)
        lSignalQuantiles = np.percentile(signal, lPercentiles)
        lSignalQuantiles = dict(zip(lPercentiles, list(lSignalQuantiles)))
        # print("SIGNAL_QUANTILES" , (self.mName , lSignalQuantiles))
        lPercentiles2 = [50 - q for q in range(Q, 50, Q)] + [50] + [50 + q for q in range(Q, 50, Q)]
        lErrorQuantiles = np.percentile(myerror, lPercentiles2)
        self.mErrorQuantiles = dict(zip(lPercentiles2, list(lErrorQuantiles)))
        return lSignalQuantiles
    
    def compute_CRPS(self, signal , estimator):
        lLossValues = []
        # some normalization
        self.mSignalQuantiles = self.compute_signal_quantiles(signal , estimator);
        for (a, q) in self.mSignalQuantiles.items():
            lDiff_q = q - estimator.values
            lPinballLoss_a = (1.0 - a / 100) * np.maximum(lDiff_q, 0.0) +  a / 100 * np.maximum(-lDiff_q, 0)
            lLossValue_a = lPinballLoss_a.mean()
            lLossValues.append(lLossValue_a)
        lCRPS = np.mean(lLossValues)
        self.mCRPS = round( lCRPS , 4 )
        # print("CRPS" , (self.mName , lCRPS))
        return self.mCRPS

    def compute_L1(self, signal , estimator):
        abs_error = self.pre_compute_abs_error_if_needed(signal , estimator);
        self.mL1 = np.mean(abs_error)
        self.mL1 = round(self.mL1, 4)
        return self.mL1
            
    def compute_MedAE(self, signal , estimator):
        abs_error = self.pre_compute_abs_error_if_needed(signal , estimator);
        self.mMedAE = np.median(abs_error)
        self.mMedAE = round(self.mMedAE, 4)
        return self.mMedAE
            
    def compute_L2(self, signal , estimator):
        abs_error = self.pre_compute_abs_error_if_needed(signal , estimator);
        self.mL2 = np.sqrt(np.mean(abs_error ** 2))            
        self.mL2 = round(self.mL2, 4)
        return self.mL2

    def get_compute_methods(self):
        lMethods = {
            "MASE": self.compute_MASE,
            "MAPE": self.compute_MAPE,
            "SMAPE": self.compute_SMAPE,
            "DiffSMAPE": self.compute_DiffSMAPE,
            "MedAE": self.compute_MedAE,
            "MAE": self.compute_L1,
            "L1": self.compute_L1,
            "RMSE": self.compute_L2,
            "L2": self.compute_L2,
            "R2": self.compute_R2,
            "PEARSONR": self.compute_pearson_r,
            "LnQ": self.compute_LnQ,
            "RMSSE": self.compute_RMSSE,
            "KS": self.compute_KS,
            "Kendalltau": self.compute_Kendall,
            "MWU": self.compute_MWU,
            "AUC": self.compute_AUC,
            "CRPS": self.compute_CRPS
        }
        return lMethods
    
    def computeCriterionValues(self, signal , estimator, criterions, name):
        self.mCachedValues = {}
        self.mName = name;
        self.mCount = signal.shape[0];
        assert(signal.shape[0] > 0)


        lMethods = self.get_compute_methods()
        lOutput = {}
        for lCriterion in criterions:
            if(lCriterion not in lMethods.keys()):
                raise tsutil.Internal_PyAF_Error("Unknown Performance Measure ['" + self.mName + "'] '" + lCriterion + "'");
            lMethod = lMethods.get(lCriterion)
            lValue = lMethod(signal, estimator)
            lOutput[lCriterion] = lValue

        
        self.mCachedValues = {}
        return lOutput

    def getCriterionValue(self, criterion):
        lDict = self.__dict__
        criterion_normalized = criterion
        if(criterion == "MAE"):
            criterion_normalized = "L1"
        if(criterion == "RMSE"):
            criterion_normalized = "L2"
        try:
            lValue = lDict.get("m" + criterion_normalized)
            return lValue
        except:
            raise tsutil.Internal_PyAF_Error("Unknown Performance Measure ['" + self.mName + "'] '" + criterion + "'");
        

    def is_acceptable_criterion_value(self, criterion, iRefValue = None):
        # percentages are bad when above 1.0
        if(criterion in ['MAPE' , 'SMAPE' , 'DiffSMAPE', 'MASE', 'CRPS']):
            lCritValue = iRefValue
            if(iRefValue is None):
                lCritValue = self.getCriterionValue(criterion)
            return (lCritValue <= 1.0)
        # otherwise, acceptable by default
        return True

    
    def is_close_criterion_value(self, criterion, value, iTolerance = 0.05, iRefValue = None):
        # percentages are close in an additive way
        lCritValue = iRefValue
        if(iRefValue is None):
            lCritValue = self.getCriterionValue(criterion)
        if(criterion in ['MAPE' , 'SMAPE', 'DiffSMAPE' , 'MASE', 'CRPS']):
            return (value <= (lCritValue + iTolerance))
        # otherwise, multiplicative
        return (value <= (lCritValue * (1.0 + iTolerance)))

    
