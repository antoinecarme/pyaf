import numpy as np
import pandas as pd
from . import SignalDecomposition_AR as tsar
import sys, os
from . import Utils as tsutil
from . import Complexity as tscomplex

import torch

def make_pytorch_reproducible(iSeed):
    torch.set_num_threads(1)
    # This ia workaround for pytorch leaking file descriptors. https://github.com/pytorch/pytorch/issues/65198
    # TODO : to be removed.
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.manual_seed(iSeed)
    torch.cuda.manual_seed(iSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(iSeed)
    os.environ['PYTHONHASHSEED'] = str(iSeed)

class cAbstract_RNN_Model(tsar.cAbstractAR):
    def __init__(self , cycle_residue_name, P , iExogenousInfo = None):
        super().__init__(cycle_residue_name, iExogenousInfo)
        self.mNbLags = P;
        self.mNbExogenousLags = P;
        sys.setrecursionlimit(1000000);
        self.set_name();
        self.mComplexity = tscomplex.eModelComplexity.High;

    def dumpCoefficients(self, iMax=10):
        logger = tsutil.get_pyaf_logger();
        logger.info("MODEL_TYPE PYTORCH")
        lSummary = [module for module in self.mModel.module.modules()][:1]
        logger.info("PYTORCH_MODEL_ARCHITECTURE " + str(lSummary))

    def build_RNN_Architecture(self, iARInputs, iARTarget):
        assert(0);

    def fit_inputs_and_targets_scalers(self, iARInputs, iARTarget):
        from sklearn.preprocessing import StandardScaler

        self.mStandardScaler_Input = StandardScaler()
        self.mStandardScaler_Target = StandardScaler()
        lARInputs = self.mStandardScaler_Input.fit_transform(iARInputs)
        lARTarget = self.mStandardScaler_Target.fit_transform(iARTarget.reshape(iARTarget.shape[0], 1))
        lARTarget = lARTarget.reshape((lARTarget.shape[0], 1))
        return (lARInputs, lARTarget)

    def get_default_pytorch_options(self):
        lDict = {}
        return lDict
    
    def get_pytorch_options(self):
        if(self.mOptions.mPytorch_Options is None):
            return self.get_default_pytorch_options()
        return self.mOptions.mPytorch_Options

    def fit_pytorch_model(self, iARInputs, iARTarget):
        lTimer = None
        if(self.mOptions.mDebug):
            lTimer = tsutil.cTimer(("TRAINING_PYTORCH_MODEL", self.mOutName))
        lOptions = self.get_pytorch_options()
        lARInputs = iARInputs.astype(np.float32)
        lARTarget = iARTarget.astype(np.float32)
        lHistory = self.mModel.fit(lARInputs, lARTarget)

    def predict_pytorch_model(self, iARInputs):
        lTimer = None
        if(self.mOptions.mDebug):
            lTimer = tsutil.cTimer(("PREDICTING_PYTORCH_MODEL", self.mOutName))
        lARInputs = self.mStandardScaler_Input.transform(iARInputs)
        lARInputs = self.reshape_inputs(lARInputs)
        lARInputs = lARInputs.astype(np.float32)
        lPredicted = self.mModel.predict(lARInputs);
        lPredicted = np.reshape(lPredicted, (-1, 1))
        lPredicted = self.mStandardScaler_Target.inverse_transform(lPredicted)
        return lPredicted


    def fit(self):
        make_pytorch_reproducible(self.mOptions.mSeed)

        series = self.mCycleResidueName; 
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        lAREstimFrame = self.mSplit.getEstimPart(self.mARFrame)

        lARInputs = lAREstimFrame[self.mInputNames].values
        lARTarget = lAREstimFrame[series].values

        (lARInputs, lARTarget) = self.fit_inputs_and_targets_scalers(lARInputs, lARTarget)

        self.build_RNN_Architecture(lARInputs, lARTarget);

        assert(lARInputs.shape[1] > 0);
        assert(lARTarget.shape[0] > 0);

        lARInputs = self.reshape_inputs(lARInputs)

        self.fit_pytorch_model(lARInputs, lARTarget)
        
        lFullARInputs = self.mARFrame[self.mInputNames].values;
        lPredicted = self.predict_pytorch_model(lFullARInputs)
        self.mARFrame[self.mOutName] = lPredicted
        
        self.compute_ar_residue(self.mARFrame)

    def transformDataset(self, df, horizon_index = 1):
        series = self.mCycleResidueName; 
        if(self.mExogenousInfo is not None):
            df = self.mExogenousInfo.transformDataset(df);
        lag_df = self.generateLagsForForecast(df);
        inputs = lag_df[self.mInputNames].values

        lPredicted = self.predict_pytorch_model(inputs)

        df[self.mOutName] = lPredicted;
        self.compute_ar_residue(df)
        return df;


class cMLP_Model(cAbstract_RNN_Model):
    def __init__(self , cycle_residue_name, P , iExogenousInfo = None):
        super().__init__(cycle_residue_name, P, iExogenousInfo)
        self.mHiddenUnits = P;

    def reshape_inputs(self, iInputs):
        return iInputs;

    def create_model(self, iNbInputs, iHidden):
        from torch import nn
        model = nn.Sequential(
            nn.Linear(iNbInputs, iHidden),
            nn.Dropout(),
            nn.Linear(iHidden, 1))
        return model.float()

    def build_RNN_Architecture(self, iARInputs, iARTarget):
        from torch import nn
        lNbLags = iARInputs.shape[1]
        lOptions = self.get_pytorch_options()
        from skorch import NeuralNetRegressor
        from skorch.callbacks import EarlyStopping
        self.mModel = NeuralNetRegressor(self.create_model(lNbLags, self.mHiddenUnits),
                                         criterion=lOptions.get("criterion", nn.MSELoss),
                                         max_epochs=lOptions.get("epochs", 20),
                                         callbacks=[EarlyStopping(patience=3)],
                                         device='cpu',
                                         verbose=0)

    def set_name(self):
        lName = "MLP" if(self.mExogenousInfo is None) else "MLPX"

        self.mFormula = lName + "(" + str(self.mNbLags) + ")";
        self.mOutName = self.mCycleResidueName +  '_' + lName + '(' + str(self.mNbLags) + ")";

class cLSTMWithOneOutput(torch.nn.Module):
    def __init__(self, iLSTM):
        super().__init__()
        self.mLSTM = iLSTM

    def forward(self, X):
        return self.mLSTM(X)[0]

class cLSTM_Model(cAbstract_RNN_Model):
    
    def __init__(self , cycle_residue_name, P , iExogenousInfo = None):
        super().__init__(cycle_residue_name, P, iExogenousInfo)
        self.mHiddenUnits = P;

    def reshape_inputs(self, iInputs):
        return iInputs.reshape(iInputs.shape[0], iInputs.shape[1]);

    def create_model(self, iNbInputs, iHidden):
        from torch import nn
        lLSTM = nn.LSTM(iNbInputs, iHidden)
        lLSTMWithOneOutput = cLSTMWithOneOutput(lLSTM)
        model = nn.Sequential(
            lLSTMWithOneOutput,
            nn.Dropout(p=0.1),
            nn.Linear(iHidden, 1))
        return model.float()

    def fit_pytorch_model(self, iARInputs, iARTarget):
        # Force some sampling for LSTM. Too slow to train.
        lTimer = None
        if(self.mOptions.mDebug):
            lTimer = tsutil.cTimer(("TRAINING_PYTORCH_MODEL", self.mOutName))
        lOptions = self.get_pytorch_options()
        lARInputs = iARInputs.astype(np.float32)
        lARTarget = iARTarget.astype(np.float32)
        lMaxSize = 1024 # Keep the last 1024 rows.
        lHistory = self.mModel.fit(lARInputs[-lMaxSize:, :], lARTarget[-lMaxSize:])        
    
    def build_RNN_Architecture(self, iARInputs, iARTarget):
        from torch import nn
        lNbLags = iARInputs.shape[1]
        lOptions = self.get_pytorch_options()
        from skorch import NeuralNetRegressor
        from skorch.callbacks import EarlyStopping
        self.mModel = NeuralNetRegressor(self.create_model(lNbLags, self.mHiddenUnits),
                                         criterion=lOptions.get("criterion", nn.MSELoss),
                                         max_epochs=lOptions.get("epochs", 10),
                                         callbacks=[EarlyStopping(patience=3)],
                                         device='cpu',
                                         verbose=0)


    def set_name(self):
        lName = "LSTM" if(self.mExogenousInfo is None) else "LSTMX"

        self.mFormula = lName + "(" + str(self.mNbLags) + ")";
        self.mOutName = self.mCycleResidueName +  '_' + lName + '(' + str(self.mNbLags) + ")";

