import numpy as np
import pandas as pd
from . import SignalDecomposition_AR as tsar
import sys


class cAbstract_RNN_Model(tsar.cAbstractAR):
    def __init__(self , cycle_residue_name, P , iExogenousInfo = None):
        super().__init__(cycle_residue_name, iExogenousInfo)
        self.mNbLags = P;
        self.mNbExogenousLags = P;
        self.mComplexity = P;
        self.mHiddenUnits = P;
        self.mNbEpochs = 100;
        sys.setrecursionlimit(1000000);

    def dumpCoefficients(self, iMax=10):
        # print(self.mModel.__dict__);
        pass

    def build_RNN_Architecture(self, iARInputs, iARTarget):
        assert(0);

    def fit_inputs_and_targets_scalers(self, iARInputs, iARTarget):
        from sklearn.preprocessing import StandardScaler

        self.mStandardScaler_Input = StandardScaler()
        self.mStandardScaler_Target = StandardScaler()
        lARInputs = self.mStandardScaler_Input.fit_transform(iARInputs)
        lARTarget = self.mStandardScaler_Target.fit_transform(iARTarget.reshape(-1, 1))
        return (lARInputs, lARTarget)

    def get_default_keras_options(self):
        lDict = {}
        return lDict
    
    def get_keras_options(self):
        if(self.mOptions.mKeras_Options is None):
            return self.get_default_keras_options()
        return self.mOptions.mKeras_Options


    def fit(self):
        import tensorflow as tf
        lOptions = self.get_keras_options()

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

        lStopCallback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=0, mode='auto')
        lHistory = self.mModel.fit(lARInputs, lARTarget,
                                   epochs=lOptions.get("epochs", 100),
                                   verbose=0, 
                                   callbacks=[lStopCallback])

        lFullARInputs = self.mARFrame[self.mInputNames].values;
        lFullARInputs = self.mStandardScaler_Input.transform(lFullARInputs)
        
        lFullARInputs = self.reshape_inputs(lFullARInputs)

        lPredicted = self.mModel.predict(lFullARInputs);
        lPredicted = np.reshape(lPredicted, (-1, 1))
        lPredicted = self.mStandardScaler_Target.inverse_transform(lPredicted)
            
        self.mARFrame[self.mOutName] = lPredicted
        self.compute_ar_residue(self.mARFrame)

    def transformDataset(self, df, horizon_index = 1):
        series = self.mCycleResidueName; 
        if(self.mExogenousInfo is not None):
            df = self.mExogenousInfo.transformDataset(df);
        lag_df = self.generateLagsForForecast(df);
        inputs = lag_df[self.mInputNames].values
        inputs = self.mStandardScaler_Input.transform(inputs)
                
        inputs = self.reshape_inputs(inputs)
        
        lPredicted = self.mModel.predict(inputs)
        lPredicted = np.reshape(lPredicted, (-1, 1))
        lPredicted = self.mStandardScaler_Target.inverse_transform(lPredicted)

        df[self.mOutName] = lPredicted;
        self.compute_ar_residue(df)
        return df;

class cMLP_Model(cAbstract_RNN_Model):
    def __init__(self , cycle_residue_name, P , iExogenousInfo = None):
        super().__init__(cycle_residue_name, P, iExogenousInfo)

    def reshape_inputs(self, iInputs):
        return iInputs;

    def build_RNN_Architecture(self, iARInputs, iARTarget):
        self.mModel = self.build_RNN_Architecture_template(iARInputs, iARTarget);

        lName = "MLP" if(self.mExogenousInfo is None) else "MLPX"

        self.mFormula = lName + "(" + str(self.mNbLags) + ")";
        self.mOutName = self.mCycleResidueName +  '_' + lName + '(' + str(self.mNbLags) + ")";

    def build_RNN_Architecture_template(self, iARInputs, iARTarget):
        lOptions = self.get_keras_options()
        import tensorflow as tf
        lNbLags = iARInputs.shape[1]

        lModel = tf.keras.Sequential()
        lModel.add(tf.keras.layers.Dense(self.mHiddenUnits, input_shape=(lNbLags,)))
        lModel.add(tf.keras.layers.Dropout(0.1))
        lModel.add(tf.keras.layers.Dense(1))
        optim = tf.keras.optimizers.Adam(learning_rate=0.1)
        lModel.compile(loss=lOptions.get("criterion", "mse"), optimizer=optim)
        return lModel;


class cLSTM_Model(cAbstract_RNN_Model):
    
    def __init__(self , cycle_residue_name, P , iExogenousInfo = None):
        super().__init__(cycle_residue_name, P, iExogenousInfo)

    def reshape_inputs(self, iInputs):
        lNewShape = (iInputs.shape[0], 1, iInputs.shape[1])
        lInputs = np.reshape(iInputs, lNewShape)
        return lInputs;


    def build_RNN_Architecture(self, iARInputs, iARTarget):
        self.mModel = self.build_RNN_Architecture_template(iARInputs, iARTarget);

        lName = "LSTM" if(self.mExogenousInfo is None) else "LSTMX"

        self.mFormula = lName + "(" + str(self.mNbLags) + ")";
        self.mOutName = self.mCycleResidueName +  '_' + lName + '(' + str(self.mNbLags) + ")";


    def build_RNN_Architecture_template(self, iARInputs, iARTarget):
        lOptions = self.get_keras_options()
        lNbLags = iARInputs.shape[1]
        import tensorflow as tf

        lModel = tf.keras.Sequential()
        lModel.add(tf.keras.layers.LSTM(self.mHiddenUnits, input_shape=(1, lNbLags)))
        lModel.add(tf.keras.layers.Dropout(0.1))
        lModel.add(tf.keras.layers.Dense(1))
        optim = tf.keras.optimizers.Adam(learning_rate=0.1)
        lModel.compile(loss=lOptions.get("criterion", "mse"), optimizer=optim)
        return lModel;

