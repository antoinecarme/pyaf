import numpy as np
import pandas as pd
from . import SignalDecomposition_AR as tsar
import sys
from . import Utils as tsutil
from . import Complexity as tscomplex


def make_tf_reproducible(iSeed):
    import tensorflow as tf
    # tf.compat.v1.disable_eager_execution()
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.random.set_seed(iSeed)
    
class cAbstract_RNN_Model(tsar.cAbstractAR):
    def __init__(self , cycle_residue_name, P , iExogenousInfo = None):
        super().__init__(cycle_residue_name, iExogenousInfo)
        self.mNbLags = P;
        self.mNbExogenousLags = P;
        self.mHiddenUnits = P;
        sys.setrecursionlimit(1000000);
        self.mComplexity = tscomplex.eModelComplexity.High;

    def dumpCoefficients(self, iMax=10):
        logger = tsutil.get_pyaf_logger();
        logger.info("MODEL_TYPE KERAS")
        logger.info("KERAS_MODEL_ARCHITECTURE " + self.mModel.to_json())

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

    def fit_keras_model(self, iARInputs, iARTarget):
        lTimer = tsutil.cTimer(("TRAINING_KERAS_MODEL", self.mOutName))
        lOptions = self.get_keras_options()
        import tensorflow as tf
        lStopCallback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, verbose=0, mode='auto')
        lHistory = self.mModel.fit(iARInputs, iARTarget,
                                   epochs=lOptions.get("epochs", 20),
                                   batch_size=None, 
                                   verbose=0, 
                                   callbacks=[])

    def predict_keras_model(self, iARInputs):
        lTimer = tsutil.cTimer(("PREDICTING_KERAS_MODEL", self.mOutName))
        lARInputs = self.mStandardScaler_Input.transform(iARInputs)
        lARInputs = self.reshape_inputs(lARInputs)
        lPredicted = self.mModel.predict(lARInputs);
        lPredicted = np.reshape(lPredicted, (-1, 1))
        lPredicted = self.mStandardScaler_Target.inverse_transform(lPredicted)
        return lPredicted

    def fit(self):
        make_tf_reproducible(self.mOptions.mSeed)
        
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

        self.fit_keras_model(lARInputs, lARTarget)

        lFullARInputs = self.mARFrame[self.mInputNames].values;

        lPredicted = self.predict_keras_model(lFullARInputs);

        self.mARFrame[self.mOutName] = lPredicted
        self.compute_ar_residue(self.mARFrame)

    def transformDataset(self, df, horizon_index = 1):
        series = self.mCycleResidueName; 
        if(self.mExogenousInfo is not None):
            df = self.mExogenousInfo.transformDataset(df);
        lag_df = self.generateLagsForForecast(df);
        inputs = lag_df[self.mInputNames].values

        lPredicted = self.predict_keras_model(inputs)

        df[self.mOutName] = lPredicted;
        self.compute_ar_residue(df)
        return df;

class cMLP_Model(cAbstract_RNN_Model):
    def __init__(self , cycle_residue_name, P , iExogenousInfo = None):
        super().__init__(cycle_residue_name, P, iExogenousInfo)

    def reshape_inputs(self, iInputs):
        return iInputs;

    def build_RNN_Architecture(self, iARInputs, iARTarget):

        lName = "MLP" if(self.mExogenousInfo is None) else "MLPX"

        self.mFormula = lName + "(" + str(self.mNbLags) + ")";
        self.mOutName = self.mCycleResidueName +  '_' + lName + '(' + str(self.mNbLags) + ")";
        self.mModel = self.build_RNN_Architecture_template(iARInputs, iARTarget);

    def build_RNN_Architecture_template(self, iARInputs, iARTarget):
        lOptions = self.get_keras_options()
        import tensorflow as tf
        lNbLags = iARInputs.shape[1]

        lModel = tf.keras.Sequential(name = "PyAF_" + self.__class__.__name__)
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

        lName = "LSTM" if(self.mExogenousInfo is None) else "LSTMX"

        self.mFormula = lName + "(" + str(self.mNbLags) + ")";
        self.mOutName = self.mCycleResidueName +  '_' + lName + '(' + str(self.mNbLags) + ")";
        self.mModel = self.build_RNN_Architecture_template(iARInputs, iARTarget);

    def build_RNN_Architecture_template(self, iARInputs, iARTarget):
        lOptions = self.get_keras_options()
        lNbLags = iARInputs.shape[1]
        import tensorflow as tf

        lModel = tf.keras.Sequential(name = "PyAF_" + self.__class__.__name__)
        lModel.add(tf.keras.layers.LSTM(self.mHiddenUnits, input_shape=(1, lNbLags)))
        lModel.add(tf.keras.layers.Dropout(0.1))
        lModel.add(tf.keras.layers.Dense(1))
        optim = tf.keras.optimizers.Adam(learning_rate=0.1)
        lModel.compile(loss=lOptions.get("criterion", "mse"), optimizer=optim)
        return lModel;

