import numpy as np
import pandas as pd
from . import SignalDecomposition_AR as tsar
import sys

import torch
from torch import nn
import torch.nn.functional as F

def make_pytorch_reproducible(iSeed):
    torch.manual_seed(iSeed)
    torch.cuda.manual_seed(iSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(iSeed)

class cAbstract_RNN_Model(tsar.cAbstractAR):
    def __init__(self , cycle_residue_name, P , iExogenousInfo = None):
        super().__init__(cycle_residue_name, iExogenousInfo)
        self.mNbLags = P;
        self.mNbExogenousLags = P;
        self.mComplexity = P;
        self.mHiddenUnits = P;
        self.mNbEpochs = 50;
        sys.setrecursionlimit(1000000);

    def dumpCoefficients(self, iMax=10):
        # print(self.mModel.__dict__);
        pass

    def build_RNN_Architecture(self, iARInputs, iARTarget):
        assert(0);

    # def reshape_inputs(self, iInputs):
        # return iInputs;

    def reshape_inputs(self, iInputs):
        lNewShape = (iInputs.shape[0], iInputs.shape[1])
        lInputs = np.reshape(iInputs, lNewShape)
        return lInputs;

    def reshape_target(self, iTarget):
        return np.reshape(iTarget, (iTarget.shape[0], 1))
    
    def fit(self):
        # print("ESTIMATE_RNN_MODEL_START" , self.mCycleResidueName);
        make_pytorch_reproducible(self.mOptions.mSeed)


        # print("ESTIMATE_RNN_MODEL_STEP1" , self.mOutName);

        series = self.mCycleResidueName; 
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        lAREstimFrame = self.mSplit.getEstimPart(self.mARFrame)

        # print("ESTIMATE_RNN_MODEL_STEP2" , self.mOutName);

        # print("mAREstimFrame columns :" , self.mAREstimFrame.columns);
        lARInputs = lAREstimFrame[self.mInputNames].values
        lARTarget = lAREstimFrame[series].values

        self.build_RNN_Architecture(lARInputs, lARTarget);

        # print(len(self.mInputNames), lARInputs.shape , lARTarget.shape)
        assert(lARInputs.shape[1] > 0);
        assert(lARTarget.shape[0] > 0);

        # print("ESTIMATE_RNN_MODEL_STEP3" , self.mOutName);

        lARInputs = self.reshape_inputs(lARInputs)
        lARTarget = self.reshape_target(lARTarget)

        N = lARInputs.shape[0];
        NEstim = (N * 4) // 5;
        estimX = lARInputs[0:NEstim]
        estimY = lARTarget[0:NEstim]
        valX = lARInputs[ NEstim : ]
        valY = lARTarget[ NEstim : ]

        # print("SHAPES" , self.mFormula, estimX.shape , estimY.shape)

        # print("ESTIMATE_RNN_MODEL_STEP4" , self.mOutName);

        lHistory = self.mModel.fit(estimX, estimY)
        
        # print(lHistory.__dict__)

        # print("ESTIMATE_RNN_MODEL_STEP5" , self.mOutName);

        lFullARInputs = self.mARFrame[self.mInputNames].values;
        lFullARInputs = self.reshape_inputs(lFullARInputs)

        # print("ESTIMATE_RNN_MODEL_STEP6" , self.mOutName);

        lPredicted = self.mModel.predict(lFullARInputs);
        # print("PREDICTED_SHAPE" , self.mARFrame.shape, lPredicted.shape);

        # print("ESTIMATE_RNN_MODEL_STEP7" , self.mOutName);
            
        self.mARFrame[self.mOutName] = np.reshape(lPredicted, (lPredicted.shape[0]))

        # print("ESTIMATE_RNN_MODEL_STEP8" , self.mOutName);

        self.compute_ar_residue(self.mARFrame)

        # print("ESTIMATE_RNN_MODEL_END" , self.mOutName, self.mModel.__dict__);
        # self.testPickle_old();

    def transformDataset(self, df, horizon_index = 1):
        series = self.mCycleResidueName; 
        if(self.mExogenousInfo is not None):
            df = self.mExogenousInfo.transformDataset(df);
        # print(df.columns);
        # print(df.info());
        # print(df.head());
        # print(df.tail());
        lag_df = self.generateLagsForForecast(df);
        # print(self.mInputNames);
        # lag_df.to_csv("LAGGED_ " + str(self.mNbLags) + ".csv");
        inputs = lag_df[self.mInputNames].values
        inputs = self.reshape_inputs(inputs)
        
        # print("BEFORE_PREDICT", self.mFormula, "\n", self.mModel.__dict__);
        lPredicted = self.mModel.predict(inputs)
        lPredicted = np.reshape(lPredicted, (lPredicted.shape[0]))
        df[self.mOutName] = lPredicted;
        self.compute_ar_residue(df)
        return df;


class cRegressorModule_MLP(nn.Module):
    def __init__(self, iNbInputs = 12, iHidden = 1):
        super().__init__()
        self.n_in = iNbInputs
        self.n_h = iHidden
        self.first_layer = nn.Linear(self.n_in, self.n_h)
        self.final_layer = nn.Linear(self.n_h, 1)
        self.double()

    def forward(self, x_batch):
        X = x_batch # .()
        X = self.first_layer(X)
        X = F.dropout(X, 0.1)
        X = self.final_layer(X)
        return X
    
class cPytorchRegressor_MLP:
    def __init__(self, iNbInputs, iHidden):
        from skorch import NeuralNetRegressor
        from torch import optim
        self.mSkorchRegressor = NeuralNetRegressor(module=cRegressorModule_MLP(iNbInputs, iHidden),
                                                   criterion=nn.MSELoss,
                                                   max_epochs=20,
                                                   device='cpu',
                                                   verbose=10)

    def fit(self, X, y):
        self.mSkorchRegressor.fit(X.astype(np.float64), y)

    def predict(self, X):
        return self.mSkorchRegressor.predict(X.astype(np.float64))

class cMLP_Model(cAbstract_RNN_Model):
    def __init__(self , cycle_residue_name, P , iExogenousInfo = None):
        super().__init__(cycle_residue_name, P, iExogenousInfo)


    def build_RNN_Architecture(self, iARInputs, iARTarget):
        lNbLags = iARInputs.shape[1]
        self.mModel =  cPytorchRegressor_MLP(lNbLags, 1)

        lName = "MLP" if(self.mExogenousInfo is None) else "MLPX"

        self.mFormula = lName + "(" + str(self.mNbLags) + ")";
        self.mOutName = self.mCycleResidueName +  '_' + lName + '(' + str(self.mNbLags) + ")";


class cRegressorModule_LSTM(nn.Module):
    def __init__(self, iNbInputs = 12, iHidden = 1):
        super().__init__()
        self.n_in = iNbInputs
        self.n_h = iHidden
        self.first_layer = nn.LSTMCell(self.n_in, self.n_h)
        self.final_layer = nn.Linear(self.n_h, 1)
        self.double()

    def forward(self, x_batch):
        X = x_batch # .()
        X = self.first_layer(X)
        X = F.dropout(X, 0.1)
        X = self.final_layer(X)
        return X
    
class cPytorchRegressor_LSTM:
    def __init__(self, iNbInputs, iHidden):
        from skorch import NeuralNetRegressor
        from torch import optim
        self.mSkorchClassifier = NeuralNetRegressor(module=cRegressorModule_LSTM(iNbInputs, iHidden),
                                                    criterion=nn.MSELoss,
                                                    max_epochs=20,
                                                    device='cpu',
                                                    verbose=10)

    def fit(self, X, y):
        self.mSkorchClassifier.fit(X.astype(np.float64), y)

    def predict(self, X):
        return self.mSkorchClassifier.predict(X.astype(np.float64))

class cLSTM_Model(cAbstract_RNN_Model):
    gTemplateModels = {};
    
    def __init__(self , cycle_residue_name, P , iExogenousInfo = None):
        super().__init__(cycle_residue_name, P, iExogenousInfo)


    def build_RNN_Architecture(self, iARInputs, iARTarget):
        lNbLags = iARInputs.shape[1]
        self.mModel =  cPytorchRegressor_MLP(lNbLags, 1)

        lName = "LSTM" if(self.mExogenousInfo is None) else "LSTMX"

        self.mFormula = lName + "(" + str(self.mNbLags) + ")";
        self.mOutName = self.mCycleResidueName +  '_' + lName + '(' + str(self.mNbLags) + ")";

