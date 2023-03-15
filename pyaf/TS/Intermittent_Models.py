import numpy as np
import pandas as pd
from . import SignalDecomposition_AR as tsar
from . import Utils as tsutil
from . import Perf as tsperf
from . import Complexity as tscomplex


def is_signal_intermittent(iSeries, iOptions):
    series = iSeries - iSeries.min()
    zero_values = series[abs(series) < 1e-8]
    lZeroRate = zero_values.shape[0] / series.shape[0]
    if(lZeroRate > iOptions.mCrostonOptions.mZeroRate):
        return True
    return False
    

class cCroston_Model(tsar.cAbstractAR):
    def __init__(self , cycle_residue_name, P , iExogenousInfo = None):
        super().__init__(cycle_residue_name, iExogenousInfo)
        self.mNbLags = 1
        self.mAlpha = None
        self.mComplexity = tscomplex.eModelComplexity.High;        

    def dumpCoefficients(self, iMax=10):
        logger = tsutil.get_pyaf_logger();
        logger.info("CROSTON_ALPHA " + str(self.mAlpha));
        logger.info("CROSTON_METHOD " + str(self.mOptions.mCrostonOptions.mMethod));
        pass

    def set_name(self):
        self.mOutName = self.mCycleResidueName +  '_CROSTON(' + str(self.mOptions.mCrostonOptions.mAlpha) + ')';
        self.mFormula = "CROSTON"

    def get_coeff(self, alpha , croston_type):
        if(croston_type == "SBA"):
            return 1.0-(alpha/2.0)
        elif(croston_type == "SBJ"):
            return (1.0 - alpha/(2.0-alpha))
        # default : any other value is the legacy croston method
        return 1.0


    def estimate_alpha(self, df):
        # print("CROSTON_OPTIONS" , self.mOptions.mCrostonOptions.__dict__)
        method = self.mOptions.mCrostonOptions.mMethod
        if(self.mOptions.mCrostonOptions.mAlpha is not None):
            self.mAlpha = self.mOptions.mCrostonOptions.mAlpha
            return 
        else:
            # choose the best alpha based on L2
            lPerfs = {}
            lForecastColumnName = 'forecast'
            for alpha in np.arange(0.05 , 1.0, 0.05):
                forecast_df = self.compute_forecast(df, alpha, method, 1)
                lPerf = tsperf.cPerf();
                lDict = lPerf.computeCriterionValues(forecast_df[self.mCycleResidueName] ,
                                                     forecast_df[lForecastColumnName] ,
                                                     [self.mOptions.mCrostonOptions.mAlphaCriterion],
                                                     "CROSTON_SEL_" + '_Fit_' + str(alpha))
                lPerfs[alpha] = lDict[self.mOptions.mCrostonOptions.mAlphaCriterion]
            self.mAlpha = min(lPerfs, key=lPerfs.get)
            # print(lPerfs)
            # print("CROSTON_OPTIMIZED_ALPHA" , self.mAlpha)
            return
    
    def croston(self, df, horizon_index = 1):
        alpha =  self.mAlpha
        method = self.mOptions.mCrostonOptions.mMethod
        df = self.compute_forecast(df, alpha , method , horizon_index)
        return df

    def simple_ses(self, x, alpha):
        #  Croston implementation is slow #182. Use statsmodels 
        if(np.std(x) < 1e-8):
            # Avoid warnings from statsmodels for constant signals 
            return x.mean() + np.zeros_like(x)
        from statsmodels.tsa.api import SimpleExpSmoothing
        lSES = SimpleExpSmoothing(x).fit(smoothing_level=alpha, optimized=False)
        y = lSES.fittedvalues
        return y

    def compute_forecast(self, df, alpha, method, horizon_index = 1):
        # print(df.shape)
        # print(df.columns)
        # print(df[['Date', 'Signal', '_Signal', 'row_number', '_Signal_ConstantTrend_residue_zeroCycle_residue']].tail(12))
        lCounts_df = df[[self.mTime, self.mCycleResidueName]].copy()
        lCounts_df['index'] = np.arange(lCounts_df.shape[0])
        df1 = lCounts_df.reset_index()

        counts = lCounts_df[self.mCycleResidueName] - self.mOffset
        counts = counts[:-(horizon_index)]
        # print(list(counts.unique()))
        # print(counts.describe())
        # assert(not np.isnan(counts[:-1]).any())
        #  q is often called the “demand” and a the “inter-arrival time”.
        q = counts[abs(counts) > 1e-8]
        if(q.shape[0] == 0):
            df1['forecast'] = self.mOffset
            return df1
        assert(q.shape[0] > 0)
        demand_times = pd.Series(list(q.index), dtype=np.float64) + 1
        a = demand_times - demand_times.shift(1).fillna(0.0)
        df2 = pd.DataFrame({'demand_time' : list(demand_times), 'q' : list(q) , 'a' : list(a) })
        
        # Use statmopdels library to perform SES to avoid recursion and also avoid reinventing the wheel.
        df2['q_est'] = self.simple_ses(df2['q'].values, alpha)
        df2['a_est'] = self.simple_ses(df2['a'].values, alpha)

        df2['forecast'] = self.get_coeff(alpha , method) * df2['q_est'] / df2['a_est']
        df2['index'] = df2['demand_time'] - 1

        for h in range(horizon_index):
            lCounts_df.loc[-(h+1), self.mCycleResidueName] = None
        df3 = df1.merge(df2 , how='left', on=('index' , 'index'))
        df4 = df3.fillna(method='ffill')
        # fill first empty fit data with zero counts (when signal starts with zeros)
        i = 0
        while(np.isnan(df4.loc[i , 'forecast'])):
            df4.loc[i , 'forecast'] = 0.0
            i = i + 1
        df4['forecast'] = df4['forecast'] + self.mOffset
        return df4
        
    def fit(self):
        #  print("ESTIMATE_CROSTON_MODEL_START" , self.mCycleResidueName);

        self.set_name();
        
        series = self.mCycleResidueName; 
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        lAREstimFrame = self.mSplit.getEstimPart(self.mARFrame)
        self.mOffset = lAREstimFrame[self.mCycleResidueName].min()
        # print("OFFSET", (self.mCycleResidueName, self.mOffset))
        self.estimate_alpha(lAREstimFrame)
        self.mFeatureSelector =  None;
        self.mInputNamesAfterSelection = self.mInputNames;

        lPredicted = self.croston(self.mARFrame);
        self.mARFrame[self.mOutName] = lPredicted['forecast']
        self.compute_ar_residue(self.mARFrame)

        # print("ESTIMATE_CROSTON_MODEL_END" , self.mOutName);


    def transformDataset(self, df, horizon_index = 1):
        series = self.mCycleResidueName;
        pred = self.croston(df, horizon_index)
        df[self.mOutName] = pred['forecast'];
        self.compute_ar_residue(df)
        return df;

