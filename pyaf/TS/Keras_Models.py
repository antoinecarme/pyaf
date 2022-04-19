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
        self.mNbEpochs = 10;
        sys.setrecursionlimit(1000000);

    def dumpCoefficients(self, iMax=10):
        # print(self.mModel.__dict__);
        pass

    def build_RNN_Architecture(self, iARInputs, iARTarget):
        assert(0);

    # def reshape_inputs(self, iInputs):
        # return iInputs;

    def reshape_inputs(self, iInputs):
        lNewShape = (iInputs.shape[0], 1, iInputs.shape[1])
        lInputs = np.reshape(iInputs, lNewShape)
        return lInputs;

    def fit(self):
        # print("ESTIMATE_RNN_MODEL_START" , self.mCycleResidueName);
        import tensorflow as tf


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

        lStopCallback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
        lHistory = self.mModel.fit(estimX, estimY,
                                   epochs=self.mNbEpochs,
                                   batch_size=1,
                                   validation_data=(valX , valY),
                                   verbose=0, 
                                   callbacks=[lStopCallback])
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

class cMLP_Model(cAbstract_RNN_Model):
    gTemplateModels = {};
    def __init__(self , cycle_residue_name, P , iExogenousInfo = None):
        super().__init__(cycle_residue_name, P, iExogenousInfo)


    def build_RNN_Architecture(self, iARInputs, iARTarget):
        lModel = None;
        lNbLags = iARInputs.shape[1]
        if(lNbLags not in cMLP_Model.gTemplateModels.keys()):
            lModel = self.build_RNN_Architecture_template(iARInputs, iARTarget);
            cMLP_Model.gTemplateModels[lNbLags] = lModel;

        import copy;
        self.mModel = copy.copy(cMLP_Model.gTemplateModels[lNbLags]);
        self.mModel.reset_states();
        # print(cMLP_Model.gTemplateModels[self.mNbLags].__dict__);
        # print(self.mModel.__dict__);

        lName = "MLP" if(self.mExogenousInfo is None) else "MLPX"

        self.mFormula = lName + "(" + str(self.mNbLags) + ")";
        self.mOutName = self.mCycleResidueName +  '_' + lName + '(' + str(self.mNbLags) + ")";

    def __getstate__(self):
        dict_out = self.__dict__.copy();
        dict_out["mModel"] = self.mModel.to_json();
        # print("GET_STATE_LSTM", dict_out);
        return dict_out;

    def __setstate__(self, istate):
        # print("LSTM_SET_STATE" , istate);
        import tensorflow as tf
        self.__dict__ = istate.copy();
        self.mModel = tf.keras.models.model_from_json(istate["mModel"]);

    def build_RNN_Architecture_template(self, iARInputs, iARTarget):
        import tensorflow as tf
        # import theano
        # print(theano.config)
        lNbLags = iARInputs.shape[1]

        lModel = tf.keras.Sequential()
        lModel.add(tf.keras.layers.Dense(self.mHiddenUnits, input_shape=(1, lNbLags)))
        lModel.add(tf.keras.layers.Dropout(0.1))
        lModel.add(tf.keras.layers.Dense(1))
        lModel.compile(loss='mse', optimizer='adam')
        return lModel;

    def reshape_target(self, iTarget):
        return np.reshape(iTarget, (iTarget.shape[0], 1, 1))


class cLSTM_Model(cAbstract_RNN_Model):
    gTemplateModels = {};
    
    def __init__(self , cycle_residue_name, P , iExogenousInfo = None):
        super().__init__(cycle_residue_name, P, iExogenousInfo)


    def build_RNN_Architecture(self, iARInputs, iARTarget):
        lModel = None;
        lNbLags = iARInputs.shape[1]
        if(lNbLags not in cLSTM_Model.gTemplateModels.keys()):
            lModel = self.build_RNN_Architecture_template(iARInputs, iARTarget);
            cLSTM_Model.gTemplateModels[lNbLags] = lModel;

        import copy;
        self.mModel = copy.copy(cLSTM_Model.gTemplateModels[lNbLags]);
        self.mModel.reset_states();
        # print(cLSTM_Model.gTemplateModels[self.mNbLags].__dict__);
        # print(self.mModel.__dict__);

        lName = "LSTM" if(self.mExogenousInfo is None) else "LSTMX"

        self.mFormula = lName + "(" + str(self.mNbLags) + ")";
        self.mOutName = self.mCycleResidueName +  '_' + lName + '(' + str(self.mNbLags) + ")";


    def build_RNN_Architecture_template(self, iARInputs, iARTarget):
        lNbLags = iARInputs.shape[1]
        import tensorflow as tf
        # import theano
        # theano.config.reoptimize_unpickled_function = False
        # theano.config.cxx = ""

        lModel = tf.keras.Sequential()
        lModel.add(tf.keras.layers.LSTM(self.mHiddenUnits, input_shape=(1, lNbLags)))
        lModel.add(tf.keras.layers.Dropout(0.1))
        lModel.add(tf.keras.layers.Dense(1))
        lModel.compile(loss='mse', optimizer='adam')
        return lModel;


    def testPickle_old(self):
        import pickle
        out1 = pickle.dumps(self.mModel);
        lModel2 = pickle.loads(out1);
        out2 = pickle.dumps(lModel2);
        print(sorted(self.mModel.__dict__))
        print(sorted(lModel2.__dict__))
        for (k , v) in self.mModel.__dict__.items():
            print(k , self.mModel.__dict__[k])
            print(k , lModel2.__dict__[k])
        assert(out1 == out2)
        print("TEST_PICKLE_OLD_OK")

    def testPickle(self):
        import dill
        dill.settings['recurse'] = False
        out1 = dill.dumps(self.mModel);
        lModel2 = dill.loads(out1);
        out2 = dill.dumps(lModel2);
        print(sorted(self.mModel.__dict__))
        print(sorted(lModel2.__dict__))
        for (k , v) in self.mModel.__dict__.items():
            print(k , self.mModel.__dict__[k])
            print(k , lModel2.__dict__[k])
        assert(out1 == out2)
        print("TEST_PICKLE_OK")

    def __getstate__(self):
        dict_out = self.__dict__.copy();
        dict_out["mModel"] = self.mModel.to_json();
        # print("GET_STATE_LSTM", dict_out);
        return dict_out;

    def __setstate__(self, istate):
        # print("LSTM_SET_STATE" , istate);
        import tensorflow as tf
        self.__dict__ = istate.copy();
        self.mModel = tf.keras.models.model_from_json(istate["mModel"]);


    def reshape_target(self, iTarget):
        return iTarget
