# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np
import datetime

from . import Utils as tsutil

class cPerf:
    def __init__(self):
        self.mErrorStdDev = None;
        self.mErrorMean = None;
        self.mMAE = None;
        self.mMAPE = None;
        self.mSMAPE = None;
        self.mL1 = None;
        self.mL2 = None;
        self.mR2 = None;
        self.mPearsonR = None;
        self.mCount = None;
        self.mName = "No_Name";

    def protect_small_value(self, x , eps):
        if((x < eps) and (x >= 0)):
            return False;
        if((x > -eps) and (x <= 0)):
            return False;
        return True;
    
    def protect_small_values(self, signal, estimator):
        eps = 1.0e-13;
        keepThis = signal.apply(lambda x : self.protect_small_value(x, eps));
        signal1 =  signal[keepThis];       
        estimator1 = estimator[keepThis];
        # self.dump_perf_data(signal , signal1);        
        return (signal1 , estimator1);

    def check_not_nan(self, sig , name):
        #print("check_not_nan");
        if(np.isnan(sig).any()):
            logger = tsutil.get_pyaf_logger();
            logger.error("PERF_WITH_NAN_IN_SIGNAL" + str(sig));
            raise tsutil.InternalForecastError("INVALID_COLUMN _FOR_PERF ['" + self.mName + "'] '" + name + "'");
        pass

    def compute_R2(self, signal , estimator):
        #return 0.0;
        SST = np.sum((signal.values - np.mean(signal.values))**2);
        SSReg = np.sum((signal.values - estimator.values)**2)
        R2 = 0;
        eps = 1.0e-10;
        if(SST < eps):
            SST = eps;
        R2 = SSReg/SST
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
            raise tsutil.InternalForecastError("Failure when computing perf ['" + self.mName + "'] '" + name + "'");
        pass
            
    def real_compute(self, signal , estimator, name):
        self.mName = name;
        self.check_not_nan(signal.values , "signal")
        self.check_not_nan(estimator.values , "estimator")

        signal_std = np.std(signal);
        estimator_std = np.std(estimator);
        
        (signal1 , estimator1) = self.protect_small_values(signal, estimator)

        myerror = (estimator.values - signal.values);
        myerror1 = (estimator1.values - signal1.values);
        abs_error = abs(myerror)
        self.mErrorMean = np.mean(myerror)
        self.mErrorStdDev = np.std(myerror)        
        self.mMAE = np.mean(abs_error)
        sum_abs = np.abs(signal1.values) + np.abs(estimator1.values)
        if(signal1.shape[0] > 0):
            self.mMAPE = np.mean(abs(myerror1 / signal1.values))
            self.mSMAPE = np.mean(abs(myerror1) / sum_abs)
            self.mMAPE = round( self.mMAPE , 4 )
            self.mSMAPE = round( self.mSMAPE , 4 )
        self.mL1 = np.mean(abs_error)
        self.mL2 = np.sqrt(np.mean(abs_error ** 2))
        self.mCount = signal.shape[0];
        self.mR2 = self.compute_R2(signal, estimator)
        self.mPearsonR = 0.0;
        
        from scipy.stats import pearsonr
        # print("PEARSONR_DETAIL1" , signal_std, estimator_std)
        # print("PEARSONR_DETAIL2" , signal)
        # print("PEARSONR_DETAIL3" , estimator)
        if((signal_std > 0.0) and (estimator_std > 0.0) and (self.mCount > 30)):
            (r , pval) = pearsonr(signal.values , estimator.values)
            self.mPearsonR = r;
        else:
            self.mPearsonR = 0.0;
#            print("COMPUTED_PERF_DETAIL " , name, self.mCount ,
#                  self.mErrorMean ,  self.mErrorStdDev ,  self.mMAE ,
#                  self.mMAPE ,  self.mSMAPE , self.mL1 ,  self.mL2 ,
#                  self.mR2 ,  self.mPearsonR);
        pass

    def computeCriterion(self, signal , estimator, criterion):
        self.mCount = signal.shape[0];

        (signal1 , estimator1) = self.protect_small_values(signal, estimator)

        myerror = (estimator.values - signal.values);
        myerror1 = (estimator1.values - signal1.values);
        abs_error = abs(myerror)
        if(criterion == "L1"):
            self.mL1 = np.mean(abs_error)
            return self.mL1;
        if(criterion == "L2"):
            self.mL2 = np.sqrt(np.mean(abs_error ** 2))
            return self.mL2;
        if(criterion == "R2"):
            self.mR2 = self.compute_R2(signal, estimator)
            return self.mR2;
        if(criterion == "PEARSONR"):
            from scipy.stats import pearsonr 
            (self.mPearsonR , pval) = pearsonr(signal1 , estimator1)
            return self.mPearsonR;
        if(criterion == "MAE"):
            self.mMAE = np.mean(abs_error)
            return self.mAE;
        if(criterion == "SMAPE"):
            if(signal1.shape[0] > 0):
                self.mSMAPE = np.mean(abs(myerror1) / sum_abs)
                self.mSMAPE = round( self.mSMAPE , 4 )
            return self.mSMAPE;
        if(criterion == "COUNT"):
            return self.mCount;
        
        if(signal1.shape[0] > 0):
            self.mMAPE = np.mean(abs(myerror1 / signal1.values))
            self.mMAPE = round( self.mMAPE , 4 )
        return self.mMAPE;

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
        if(criterion == "COUNT"):
            return self.mCount;
        return self.mMAPE;

        
#def date_to_number(x):
#    return  date(int(x) , int(12 * (x - int(x) + 0.01)) + 1 , 1)
