import pandas as pd
import numpy as np
import datetime
from scipy.stats import pearsonr 

from . import SignalDecomposition_utils as tsutil

class cPerf:
    def __init__(self):
        self.mErrorStdDev = np.nan;
        self.mErrorMean = np.nan;
        self.mMAE = np.nan;
        self.mMAPE = np.nan;
        self.mSMAPE = np.nan;
        self.mL1 = np.nan;
        self.mL2 = np.nan;
        self.mR2 = np.nan;
        self.mPearsonR = np.nan;
        self.mCount = np.nan;
        self.mName = "No_Name";

    def protect_small_values(self, signal):
        eps = 1.0e-10;
        signal1 = signal.copy()
        signal1[signal1 < eps][signal1 >= 0] = eps
        signal1[signal1 > -eps][signal1 <= 0] = -eps
        return signal1;

    def check_not_nan(self, sig , name):
        #print("check_not_nan");
        if(np.isnan(sig).any()):
            print("PERF_WITH_NAN_IN_SIGNAL" , sig);
            raise tsutil.InternalForecastError("Invalid column for perf ['" + self.mName + "'] '" + name + "'");
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
        df = pd.DataFrame();
        df['sig'] = signal.values;
        df['est'] = estimator.values;
        print(df.head());
        print(df.tail());
    
    def compute(self, signal , estimator, name):
        try:
            # self.dump_perf_data(signal, estimator);
            return self.real_compute(signal, estimator, name);
        except:
            self.dump_perf_data(signal, estimator);
            raise tsutil.InternalForecastError("Failure when computing perf ['" + self.mName + "'] '" + name + "'");
        pass
            
    def real_compute(self, signal , estimator, name):
        self.mName = name;
        self.check_not_nan(signal.values , "signal")
        self.check_not_nan(estimator.values , "estimator")

        signal_std = np.std(signal);
        estimator_std = np.std(estimator);
        
        signal1 = self.protect_small_values(signal)
        estimator1 = self.protect_small_values(estimator);

        myerror = (estimator.values - signal.values);
        abs_error = abs(myerror)
        self.mErrorMean = np.mean(myerror)
        self.mErrorStdDev = np.std(myerror)        
        self.mMAE = np.mean(abs_error)
        sum_abs = np.abs(signal1.values) + np.abs(estimator1.values)
        self.mMAPE = np.mean(abs(myerror / signal1.values))
        self.mSMAPE = np.mean(abs(myerror) / sum_abs)
        self.mMAPE = round( self.mMAPE , 2 )
        self.mSMAPE = round( self.mSMAPE , 2 )
        self.mL1 = np.mean(abs_error)
        self.mL2 = np.sqrt(np.mean(abs_error ** 2))
        self.mCount = signal.shape[0];
        self.mR2 = self.compute_R2(signal1, estimator1)
        self.mPearsonR = 0.0;
        if((signal_std > 0.0) and (estimator_std > 0.0)):
            (r , pval) = pearsonr(signal1 , estimator1)
            self.mPearsonR = r;
#            print("COMPUTED_PERF_DETAIL " , name, self.mCount ,
#                  self.mErrorMean ,  self.mErrorStdDev ,  self.mMAE ,
#                  self.mMAPE ,  self.mSMAPE , self.mL1 ,  self.mL2 ,
#                  self.mR2 ,  self.mPearsonR);
        pass

    def computeCriterion(self, signal , estimator, criterion):
        self.mCount = signal.shape[0];

        signal1 = self.protect_small_values(signal)
        estimator1 = self.protect_small_values(estimator);

        myerror = (estimator.values - signal.values);
        abs_error = abs(myerror)
        if(criterion == "L1"):
            self.mL1 = np.mean(abs_error)
            return self.mL1;
        if(criterion == "L2"):
            self.mL2 = np.sqrt(np.mean(abs_error ** 2))
            return self.mL2;
        if(criterion == "R2"):
            self.mR2 = self.compute_R2(signal1, estimator1)
            return self.mR2;
        if(criterion == "PEARSONR"):
            (self.mPearsonR , pval) = pearsonr(signal1 , estimator1)
            return self.mPearsonR;
        if(criterion == "MAE"):
            self.mMAE = np.mean(abs_error)
            return self.mAE;
        if(criterion == "SMAPE"):
            self.mSMAPE = np.mean(abs(myerror) / sum_abs)
            self.mSMAPE = round( self.mSMAPE , 2 )
            return self.mSMAPE;
        if(criterion == "COUNT"):
            return self.mCount;
        
        self.mMAPE = np.mean(abs(myerror / signal1.values))
        self.mMAPE = round( self.mMAPE , 2 )
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
