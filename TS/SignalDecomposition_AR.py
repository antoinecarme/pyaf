import pandas as pd
import numpy as np
import datetime

# from memory_profiler import profile

from . import Time as tsti
from . import Perf as tsperf
from . import Plots as tsplot


# for timing
import time

class cAbstractAR:
    def __init__(self , cycle_residue_name):
        self.mTimeInfo = tsti.cTimeInfo()
        self.mCycleFrame = pd.DataFrame()
        self.mARFrame = pd.DataFrame()        
        self.mCycleResidueName = cycle_residue_name
        self.mComplexity = None;
        self.mFormula = None;
        self.mTargetName = self.mCycleResidueName;
        self.mInputNames = None;
        self.mExogenousInfo = None;

    def plot(self):
        tsplot.decomp_plot(self.mARFrame, self.mTimeInfo.mNormalizedTimeColumn,
                           self.mCycleResidueName, self.mOutName , self.mOutName + '_residue');

    def dumpCoefficients(self):
        pass

    def computePerf(self):
        self.mARFitPerf= tsperf.cPerf();
        self.mARForecastPerf= tsperf.cPerf();
        (lFrameFit, lFrameForecast, lFrameTest) = self.mTimeInfo.cutFrame(self.mARFrame);
        self.mARFitPerf.compute(
            lFrameFit[self.mCycleResidueName], lFrameFit[self.mOutName], self.mOutName)
        self.mARForecastPerf.compute(
            lFrameForecast[self.mCycleResidueName], lFrameForecast[self.mOutName], self.mOutName)

class cZeroAR(cAbstractAR):
    def __init__(self , cycle_residue_name):
        super().__init__(cycle_residue_name)
        self.mOutName = self.mCycleResidueName +  '_NoAR'
        self.mNbLags = 0;
        self.mFormula = "NoAR";
        self.mComplexity = 0;
        
    def fit(self):
        series = self.mCycleResidueName; 
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        # self.mTimeInfo.addVars(self.mARFrame);
        # self.mARFrame[series] = self.mCycleFrame[series]
        self.mARFrame[self.mOutName] = self.mARFrame[series] * 0.0;
        self.mARFrame[self.mOutName + '_residue'] = self.mARFrame[series];
                

    def transformDataset(self, df):
        series = self.mCycleResidueName; 
        df[self.mOutName] = 0.0;
        target = df[series].values
        df[self.mOutName + '_residue'] = target - df[self.mOutName].values        
        return df;


class cAutoRegressiveModel(cAbstractAR):
    def __init__(self , cycle_residue_name, P, iExogenousInfo = None ):
        super().__init__(cycle_residue_name)
        self.mNbLags = P;
        self.mExogenousInfo = iExogenousInfo;
        self.mDefaultValues = {};
        self.mLagOrigins = {};
        self.mComplexity = P;

    def getDefaultValue(self, lag):
        return self.mDefaultValues[lag];

    def addLagForForecast(self, df, lag_df, series, p):
        name = series+'_Lag' + str(p);
        lSeries = df[series];
        lShiftedSeries = lSeries.shift(p)
        lDefaultValue = self.mDefaultValues[series];
            
        for i in range(p):
            lShiftedSeries.iloc[ i ] = lDefaultValue;
            
        lag_df[name] = lShiftedSeries;
        
    def generateLagsForForecast(self, df, P):
        lag_df = pd.DataFrame()
        lag_df[self.mCycleResidueName] = df[self.mCycleResidueName]
        for p in range(1,P+1):
            # signal lags ... plain old AR model
            self.addLagForForecast(df, lag_df, self.mCycleResidueName, p);
            # Exogenous variables lags
            if(self.mExogenousInfo is not None):
                # print(self.mExogenousInfo.mEncodedExogenous);
                # print(df.columns);
                for ex in self.mExogenousInfo.mEncodedExogenous:
                    self.addLagForForecast(df, lag_df, ex, p);
        return lag_df;

    def dumpCoefficients(self):
        lDict = dict(zip(self.mInputNames , self.mARRidge.coef_));
        i = 1;
        for k in sorted(lDict.keys()):
            print("AR_MODEL_COEFF" , i, k , lDict[k]);
            i = i + 1;

    
    def fit(self):
        import sklearn.linear_model as linear_model
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_regression
        self.mARRidge = linear_model.Ridge()

        series = self.mCycleResidueName; 
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mOutName = self.mCycleResidueName +  '_AR(' + str(self.mNbLags) + ")";
        self.mFormula = "AR(" + str(self.mNbLags) + ")";
        if(self.mExogenousInfo is not None):
            self.mOutName = self.mCycleResidueName +  '_ARX(' + str(self.mNbLags) + ")";
            self.mFormula = "ARX(" + str(self.mNbLags) + ")";
        lAREstimFrame = self.mTimeInfo.getEstimPart(self.mARFrame)

        # print("mAREstimFrame columns :" , self.mAREstimFrame.columns);
        lARInputs = lAREstimFrame[self.mInputNames].values
        lARTarget = lAREstimFrame[series].values
        lMaxFeatures = self.mOptions.mMaxFeatrureForAutoreg;
        if(lMaxFeatures >= lARInputs.shape[1]):
            lMaxFeatures = lARInputs.shape[1];
        self.mFeatureSelector =  SelectKBest(f_regression, k= lMaxFeatures);
        self.mFeatureSelector.fit(lARInputs, lARTarget);
        lARInputsAfterSelection =  self.mFeatureSelector.transform(lARInputs);
        del lARInputs;
        # print("FEATURE_SELECTION" , self.mOutName, lARInputs.shape[1] , lARInputsAfterSelection.shape[1]);
        self.mARRidge.fit(lARInputsAfterSelection, lARTarget)
        del lARInputsAfterSelection;
        del lARTarget;
        del lAREstimFrame;        
        
        lARInputsFull = self.mFeatureSelector.transform(self.mARFrame[self.mInputNames].values)
        self.mARFrame[self.mOutName] = self.mARRidge.predict(lARInputsFull)
        self.mARFrame[self.mOutName + '_residue'] =  self.mARFrame[series] - self.mARFrame[self.mOutName]

    def transformDataset(self, df):
        series = self.mCycleResidueName; 
        if(self.mExogenousInfo is not None):
            df = self.mExogenousInfo.transformDataset(df);
        # print(df.columns);
        lag_df = self.generateLagsForForecast(df, self.mNbLags);
        # print(self.mInputNames);
        # print(self.mFormula, "\n", lag_df.columns);
        # lag_df.to_csv("LAGGED_ " + str(self.mNbLags) + ".csv");
        inputs = lag_df[self.mInputNames].values
        inputs_after_feat_selection = self.mFeatureSelector.transform(inputs);
        pred = self.mARRidge.predict(inputs_after_feat_selection)
        df[self.mOutName] = pred;
        target = df[series].values
        df[self.mOutName + '_residue'] = target - df[self.mOutName].values        
        return df;
        

class cAutoRegressiveEstimator:
    def __init__(self):
        self.mTimeInfo = tsti.cTimeInfo()
        self.mCycleFrame = pd.DataFrame()
        self.mARFrame = pd.DataFrame()
        self.mARList = {}
        self.mExogenousInfo = None;
        
    def plotAR(self):
        for trend in self.mTrendList:
            for cycle in self.mCycleList[trend]:
                cycle_residue = cycle.getCycleResidueName();
                for autoreg in self.mARList[cycle_residue]:
                    autoreg.plot(); 

    def addLagForTraining(self, df, lag_df, series, autoreg, p):
        name = series+'_Lag' + str(p);
        autoreg.mInputNames.append(name);
        if(name in lag_df.columns):
            return lag_df;

        lSeries = df[series];
        lShiftedSeries = lSeries.shift(p)
        lDefaultValue = lSeries.iloc[0];
        if(series in self.mDefaultValues.keys()):
            lDefaultValue = self.mDefaultValues[series];
        else:
            self.mDefaultValues[series] = lDefaultValue;
            
        for i in range(p):
            lShiftedSeries.iloc[ i ] = lDefaultValue;
            
        lag_df[name] = lShiftedSeries;
        self.mLagOrigins[name] = series;
        return lag_df;

    def addLagsForTraining(self, df, cycle_residue, iHasARX = False):
        add_lag_start_time = time.time()
        for autoreg in self.mARList[cycle_residue]:
            autoreg.mInputNames = [];
            P = autoreg.mNbLags;
            for p in range(1,P+1):
                # signal lags ... plain old AR model
                self.addLagForTraining(df, self.mARFrame, cycle_residue, autoreg, p);
                # Exogenous variables lags
                if(autoreg.mExogenousInfo is not None):
                    # print(self.mExogenousInfo.mEncodedExogenous);
                    # print(df.columns);
                    for ex in self.mExogenousInfo.mEncodedExogenous:
                        self.addLagForTraining(df, self.mARFrame, ex, autoreg, p);
            # print("AUTOREG_DETAIL" , P , len(autoreg.mInputNames));
            if(autoreg.mExogenousInfo is not None):
                assert((P + P*len(self.mExogenousInfo.mEncodedExogenous)) == len(autoreg.mInputNames));
            else:
                assert(P == len(autoreg.mInputNames));
        if(self.mOptions.mDebugProfile):
            print("LAG_TIME_IN_SECONDS " + self.mTimeInfo.mSignal + " " +
                  str(len(self.mARFrame.columns)) + " " +
                  str(time.time() - add_lag_start_time))

    # @profile
    def estimate_ar_models_for_cycle(self, cycle_residue):
        self.mARFrame = pd.DataFrame();
        self.mTimeInfo.addVars(self.mARFrame);
        self.mARFrame[cycle_residue] = self.mCycleFrame[cycle_residue]            

        self.mDefaultValues = {};
        self.mLagOrigins = {};

        if(self.mOptions.mDebugProfile):
            print("AR_MODEL_ADD_LAGS_START '" +
                  cycle_residue + "' " + str(self.mCycleFrame.shape[0]) + " "
                  + str(self.mARFrame.shape[1]));

        self.addLagsForTraining(self.mCycleFrame, cycle_residue);

        if(self.mOptions.mDebugProfile):
            print("AR_MODEL_ADD_LAGS_END '" +
                  cycle_residue + "' " + str(self.mCycleFrame.shape[0]) + " "
                  + str(self.mARFrame.shape[1]));

        # print(list(self.mARFrame.columns));
        
        for autoreg in self.mARList[cycle_residue]:
            start_time = time.time()
            if(self.mOptions.mDebugProfile):
                print("AR_MODEL_START_TRAINING_TIME '" +
                      cycle_residue + "' " + str(self.mCycleFrame.shape[0]) +
                      " " +  str(len(autoreg.mInputNames)) + " " + str(start_time));
            autoreg.mOptions = self.mOptions;
            autoreg.mCycleFrame = self.mCycleFrame;
            autoreg.mARFrame = self.mARFrame;
            autoreg.mTimeInfo = self.mTimeInfo;
            autoreg.mLagOrigins = self.mLagOrigins;
            autoreg.mDefaultValues = self.mDefaultValues;
            autoreg.fit();
            autoreg.computePerf();
            end_time = time.time()
            lTrainingTime = round(end_time - start_time , 2);
            if(self.mOptions.mDebugProfile):
                print("AR_MODEL_TRAINING_TIME_IN_SECONDS '" +
                      autoreg.mOutName + "' " + str(self.mCycleFrame.shape[0]) +
                      " " +  str(len(autoreg.mInputNames)) + " " + str(lTrainingTime));
        
    # @profile
    def estimate(self):
        mARList = {}
        for trend in self.mTrendList:
            for cycle in self.mCycleList[trend]:
                cycle_residue = cycle.getCycleResidueName();
                self.mARList[cycle_residue] = [ cZeroAR(cycle_residue)];
                lHasARX = False;
                if(self.mOptions.mEnableARModels or self.mOptions.mEnableARXModels):
                    if((self.mCycleFrame[cycle_residue].shape[0] > 12) and (self.mCycleFrame[cycle_residue].std() > 0.00001)):
                        lLags = self.mCycleFrame[cycle_residue].shape[0] // 4;
                        if(lLags >= self.mOptions.mMaxAROrder):
                            lLags = self.mOptions.mMaxAROrder;
                        if(self.mOptions.mEnableARModels):
                            lAR = cAutoRegressiveModel(cycle_residue, lLags);
                            self.mARList[cycle_residue] = self.mARList[cycle_residue] + [lAR];
                        if(self.mOptions.mEnableARXModels and
                           (self.mExogenousInfo is not None)):
                            lARX = cAutoRegressiveModel(cycle_residue, lLags,
                                                        self.mExogenousInfo);
                            self.mARList[cycle_residue] = self.mARList[cycle_residue] + [lARX];
                            lHasARX = True;

        if(lHasARX):
            if(self.mOptions.mDebugProfile):
                print("AR_MODEL_ADD_EXOGENOUS '" + str(self.mCycleFrame.shape[0]) +
                      " " + str(len(self.mExogenousInfo.mEncodedExogenous)));
            self.mCycleFrame = self.mExogenousInfo.transformDataset(self.mCycleFrame);
        
        for cycle_residue in self.mARList.keys():
            self.estimate_ar_models_for_cycle(cycle_residue);
            for autoreg in self.mARList[cycle_residue]:
                autoreg.mARFrame = pd.DataFrame();
            del self.mARFrame;
