# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np

from . import Utils as tsutil

class cPerf:
    def __init__(self):
        self.mErrorStdDev = None;
        self.mErrorMean = None;
        self.mMAE = None;
        self.mMAPE = None;
        self.mSMAPE = None;
        self.mMASE = None;
        self.mL1 = None;
        self.mL2 = None;
        self.mR2 = None;
        self.mPearsonR = None;
        self.mCount = None;
        self.mName = "No_Name";
        self.mDebug = False;

    def check_not_nan(self, sig , name):
        #print("check_not_nan");
        if(np.isnan(sig).any()):
            logger = tsutil.get_pyaf_logger();
            logger.error("PERF_WITH_NAN_IN_SIGNAL" + str(sig));
            raise tsutil.Internal_PyAF_Error("INVALID_COLUMN _FOR_PERF ['" + self.mName + "'] '" + name + "'");
        pass

    def compute_MAPE_SMAPE_MASE(self, signal , estimator):
        self.mMAPE = None;
        self.mSMAPE = None;
        self.mMASE = None;
        if(signal.shape[0] > 0):
            lEps = 1.0e-10;
            abs_error = np.abs(estimator.values - signal.values);
            sum_abs = np.abs(signal.values) + np.abs(estimator.values) + lEps
            abs_rel_error = abs_error / (np.abs(signal) + lEps)
            signal_diff = signal - signal.shift(1)
            self.mMAPE = np.mean(abs_rel_error)
            self.mSMAPE = np.mean(2.0 * abs_error / sum_abs)
            if(signal_diff.shape[0] > 1):
                mean_dev_signal = np.mean(abs(signal_diff.values[1:])) + lEps;
                self.mMASE = np.mean(abs_error / mean_dev_signal)
                self.mMASE = round( self.mMASE , 4 )
            self.mMAPE = round( self.mMAPE , 4 )
            self.mSMAPE = round( self.mSMAPE , 4 )

    def compute_R2(self, signal , estimator):
        SST = np.sum((signal.values - np.mean(signal.values))**2) + 1.0e-10;
        SSRes = np.sum((signal.values - estimator.values)**2)
        R2 = 1 - SSRes/SST
        return R2

    def dump_perf_data(self, signal , estimator):
        logger = tsutil.get_pyaf_logger();
        df = pd.DataFrame();
        df['sig'] = signal.values;
        df['est'] = estimator.values;
        logger.debug(str(df.head()));
        logger.debug(str(df.tail()));
    
    def compute(self, signal , estimator, name):
        try:
            # self.dump_perf_data(signal, estimator);
            return self.real_compute(signal, estimator, name);
        except:
            self.dump_perf_data(signal, estimator);
            logger = tsutil.get_pyaf_logger();
            logger.error("Failure when computing perf ['" + self.mName + "'] '" + name + "'");
            raise tsutil.Internal_PyAF_Error("Failure when computing perf ['" + self.mName + "'] '" + name + "'");
        pass

    def compute_pearson_r(self, signal , estimator):
        from scipy.stats import pearsonr
        # print("PEARSONR_DETAIL1" , signal_std, estimator_std)
        # print("PEARSONR_DETAIL2" , signal)
        # print("PEARSONR_DETAIL3" , estimator)
        signal_std = np.std(signal);
        estimator_std = np.std(estimator);

        r = 0.0;
        if((signal_std > 0.0) and (estimator_std > 0.0) and (signal.shape[0] > 30)):
            (r , pval) = pearsonr(signal.values , estimator.values)
        return r;
        
            
    def real_compute(self, signal , estimator, name):
        self.mName = name;
        if(self.mDebug):
            self.check_not_nan(signal.values , "signal")
            self.check_not_nan(estimator.values , "estimator")

        signal_std = np.std(signal);
        estimator_std = np.std(estimator);

        self.compute_MAPE_SMAPE_MASE(signal, estimator);

        myerror = (estimator.values - signal.values);
        abs_error = abs(myerror)
        self.mErrorMean = np.mean(myerror)
        self.mErrorStdDev = np.std(myerror)        
        self.mMAE = np.mean(abs_error)
        
        self.mL1 = np.mean(abs_error)
        self.mL2 = np.sqrt(np.mean(abs_error ** 2))
        self.mCount = signal.shape[0];
        self.mR2 = self.compute_R2(signal, estimator)
        
        self.mPearsonR = self.compute_pearson_r(signal , estimator);
        

    def computeCriterion(self, signal , estimator, criterion, name):
        self.mName = name;
        
        self.mCount = signal.shape[0];
        if(criterion == "L1"):
            myerror = (estimator.values - signal.values);
            abs_error = abs(myerror)
            self.mL1 = np.mean(abs_error)
            return self.mL1;
        if(criterion == "L2"):
            myerror = (estimator.values - signal.values);
            self.mL2 = np.sqrt(np.mean(myerror ** 2))
            return self.mL2;
        if(criterion == "R2"):
            self.mR2 = self.compute_R2(signal, estimator)
            return self.mR2;
        if(criterion == "PEARSONR"):
            self.mPearsonR = self.compute_pearson_r(signal , estimator)
            return self.mPearsonR;
        if(criterion == "MAE"):
            myerror = (estimator.values - signal.values);
            abs_error = abs(myerror)
            self.mMAE = np.mean(abs_error)
            return self.mAE;
        
        if(criterion == "MAPE"):
            self.compute_MAPE_SMAPE_MASE(signal , estimator);
            return self.mMAPE;

        if(criterion == "SMAPE"):
            self.compute_MAPE_SMAPE_MASE(signal , estimator);
            return self.mSMAPE;

        if(criterion == "MASE"):
            self.compute_MAPE_SMAPE_MASE(signal , estimator);
            return self.mMASE;

        if(criterion == "COUNT"):
            return self.mCount;
        
        assert(0)
        return 0.0;

    def getCriterionValue(self, criterion):
        if(criterion == "L1"):
            return self.mL1;
        if(criterion == "L2"):
            return self.mL2;
        if(criterion == "R2"):
            return self.mR2;
        if(criterion == "PEARSONR"):
            return self.mPearsonR;
        if(criterion == "MAE"):
            return self.mAE;
        if(criterion == "SMAPE"):
            return self.mSMAPE;
        if(criterion == "MAPE"):
            return self.mMAPE;
        if(criterion == "MASE"):
            return self.mMASE;
        if(criterion == "COUNT"):
            return self.mCount;
        assert(0)
        return 0.0;

        
#def date_to_number(x):
#    return  date(int(x) , int(12 * (x - int(x) + 0.01)) + 1 , 1)
