import numpy as np
import pandas as pd
from . import SignalDecomposition_AR as tsar
from . import Utils as tsutil
import os
from . import Utils as tsutil

class cR_Model(tsar.cAbstractAR):
    def __init__(self , cycle_residue_name, P , iExogenousInfo = None):
        super().__init__(cycle_residue_name, iExogenousInfo)
        self.mNbLags = P;
        self.mNbExogenousLags = P;
        self.mComplexity = P;
        self.mDebugMode = False
        self.mDirName = "/tmp/pyaf_prototyping"

        
    def create_subproces_if_needed(self):
        import pexpect
        lR_Process = pexpect.spawn("Rscript -")
        return lR_Process

    def add_logging(self, logname):
        lScript = "options(warn=1);\n"
        lScript = lScript + 'sink(file("' + logname + '.log" , open="wt"), type="output");\n'
        lScript = lScript + 'sink(file("' + logname + '.err" , open="wt"), type="message");\n'
        return lScript

    def dump_log_and_error(self, filename):
        if(not self.mDebugMode):
            return
        print("EXEC_R_SCRIPT_ERR_START" , filename + ".err")
        self.dump_output_file(filename + ".err" , "EXEC_R_SCRIPT_ERR_DETAIL")
        print("EXEC_R_SCRIPT_ERR_END" , filename + ".err")
        print("EXEC_R_SCRIPT_LOG_START" , filename + ".log")
        self.dump_output_file(filename + ".log" , "EXEC_R_SCRIPT_LOG_DETAIL")
        print("EXEC_R_SCRIPT_LOG_END" , filename + ".log")

    def dump_output_file(self, filename , iPrefix):
        lines = open(filename, "r").readlines()
        for (idx , line) in enumerate(lines):
            print(iPrefix, idx , line[:-1][:100])
            
    def execute_r_script(self, iScript, filename):
        lTimer = None
        if(self.mDebugMode):
            lTimer = tsutil.cTimer(("PYAF_CARET_EXECUTE_R_SCRIPT", filename))
        lR_Process = self.create_subproces_if_needed()
        lScript = "write('', \"" + filename + ".lock\")\n\n"
        lScript = lScript + self.add_logging(filename)
        lScript = lScript + iScript
        lScript = lScript + "\nfile.remove(\"" + filename + ".lock\")\n\n"
        lScript = lScript + "sink(type=\"output\");\n"
        lScript = lScript + "sink(type=\"message\");\n"
        lScript = lScript + "print('end')\n"
        train_file = open(filename + ".R", "w");
        train_file.write(lScript);
        train_file.close();
        line = "source(\"" + filename + ".R\")"
        if(self.mDebugMode):
            print("EXECUTE_R_SCRIPT_DETAIL", (line))
        return_code = 0
        try:
            lR_Process.sendline(line + "\n")
            lR_Process.expect('"end"', timeout = 1200)
        except Exception as e:
            if(self.mDebugMode):
                print(("R_PROCESS_EXCEPTION", str(e)))
            return_code = 1
        self.dump_log_and_error(filename)
        return return_code

    def generate_temp_name(self):
        # lHexRef = str(hex(id(self.mContext)))
        from datetime import datetime
        d = datetime.now()
        lPrefix = d.strftime("%Y%m%d%H%M%S.%f") + "_" + str(id(self))
        return lPrefix

    def create_dirs(self):
        try:
            if(self.mDebugMode):
                print("CREATING_DIRECTORY" , self.mDirName)
            os.mkdir(self.mDirName)
        except:
            pass

        try:
            if(self.mDebugMode):
                print("CREATING_DIRECTORY" , self.mModelName)
            os.mkdir(self.mModelName)
        except:
            pass

    def save_dataset(self, name, data):
        lTimer = None
        if(self.mDebugMode):
            lTimer = tsutil.cTimer(("PYAF_CARET_SAVE_DATAESET_", name))
        self.create_dirs()
        (X, y) = data
        df = pd.DataFrame()
        if(not self.is_autoregressive_model()):
            if(len(X.shape) == 2):
                NC = X.shape[1]
                # use Feature_0, ... for naming th e features, lag names may be too complex for csv parsing 
                df = pd.DataFrame(X)
                df.columns =  ['Feature_' + str(c) for c in range(NC)]
        if(y is not None):
            df['TGT'] = y
        import csv
        df.to_csv(self.mModelName + "/" + name, index=False, quoting=csv.QUOTE_NONNUMERIC)
        return df

    def add_needed_library(self, name):
        lScript = 'library(' + name + ', quietly = TRUE);\n'
        lScript = lScript + 'cat("R_PACKAGE_VERSION",  "' + name + '", toString(packageVersion("' + name + '")) , "\\n");\n'
        return lScript

    def create_rds_model_if_needed(self):
        serialized_model_file = self.mModelName + '/model.rds'
        if(os.path.isfile(serialized_model_file)):
            return
        lSerializedData = self.mTrainingResults.get('serialized_model')
        if(lSerializedData is None):
            raise Exception("PREDICT_FAILED_MODEL_NOT_TRAINED '"  + self.mModelName + "'")           
        with open(serialized_model_file, mode='wb') as saved_model_file:
           saved_model_file.write(lSerializedData)

    def is_autoregressive_model(self):
        # TAR models are autoregressive (input = signal), while MARS are regression models (input = lags) 
        return False

    def fit(self):
        self.set_name();
        lTimer = None
        if(self.mDebugMode):
            lTimer = tsutil.cTimer(("PYAF_CARET_FIT", self.mModelName))

        self.mTrainingResults = {}
        series = self.mCycleResidueName; 
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;        
        lAREstimFrame = self.mSplit.getEstimPart(self.mARFrame)

        lARInputs = lAREstimFrame[self.mInputNames].values
        lARTarget = lAREstimFrame[series].values
        if(self.is_autoregressive_model()):
            lARInputs = None

        self.create_dirs()
        lCSVName = "training.csv"
        df = self.save_dataset(lCSVName, (lARInputs , lARTarget))
        
        lScript = ""
        lScript = lScript + 'set.seed(' + str(self.mOptions.mSeed) + ')\n'
        lScript = lScript + 'paste("R_VERSION" , R.version.string)\n'
        lScript = lScript + 'df = read.csv("' + self.mModelName + "/" + lCSVName+ '", header=TRUE)\n'
        lScript = lScript + self.add_specific_train_code()
        lScript = lScript + 'saveRDS(model, "' + self.mModelName + '/model.rds")\n'
        self.mTrainingResults['script'] = lScript
        filename = self.mModelName + "/train"
        if(self.mDebugMode):
            print("EXECUTING_R_TRAIN_SCRIPT_START" , self.mMethod , filename)
        res = self.execute_r_script(lScript, filename)
        if(self.mDebugMode):
            print("EXECUTING_R_TRAIN_SCRIPT_END" , self.mMethod , filename, res)
        if(str(res) != "0"):
            raise Exception("TRAIN_FAILED '"  + self.mModelName + "'")

        with open(self.mModelName + '/model.rds', mode='rb') as saved_model_file:
            self.mTrainingResults['serialized_model'] = saved_model_file.read()
        
        self.mTrainingResults['script'] = lScript
        lFullARInputs = self.mARFrame[self.mInputNames].values;
        if(self.is_autoregressive_model()):
            lFullARInputs = self.mARFrame[series].values
        lPredicted = self.predict(lFullARInputs);
        self.mARFrame[self.mOutName] = lPredicted[lPredicted.columns[1]]

        self.compute_ar_residue(self.mARFrame)

    def predict(self, X):
        lTimer = None
        if(self.mDebugMode):
            lTimer = tsutil.cTimer(("PYAF_CARET_PREDICT", self.mModelName))
        self.create_rds_model_if_needed()
        lTmpName = self.generate_temp_name()
        prefix = "predict_" + lTmpName
        lCSVNameIn =  prefix  + "_input.csv"
        lCSVNameOut = prefix + "_output.csv"
        if(not self.is_autoregressive_model()):
            df = self.save_dataset(lCSVNameIn , (X , None))
        else:
            df = self.save_dataset(lCSVNameIn , (None, X))
            
        lScript = ""
        lScript = lScript + 'paste("R_VERSION" , R.version.string)\n'
        lScript = lScript + 'df = read.csv("' + self.mModelName + "/" + lCSVNameIn + '", header=TRUE)\n'
        lScript = lScript + 'reloaded_model = readRDS("' + self.mModelName + '/model.rds")\n'
        lScript = lScript + self.add_specific_predict_code(X)
        lScript = lScript + 'write.csv(predicted, file = "' + self.mModelName + '/' + lCSVNameOut + '")\n'
        filename = self.mModelName + "/" + prefix
        if(self.mDebugMode):
            print("EXECUTING_R_PREDICT_SCRIPT_START" , self.mMethod , filename)
        res = self.execute_r_script(lScript, filename)
        if(self.mDebugMode):
            print("EXECUTING_R_PREDICT_SCRIPT_END" , self.mMethod , filename , str(res))
        if(str(res) != "0"):
            raise Exception("PREDICT_FAILED '"  + self.mModelName + "'")
        df = pd.read_csv(self.mModelName + "/" + lCSVNameOut)
        return df
            

    def transformDataset(self, df, horizon_index = 1):
        series = self.mCycleResidueName; 
        if(self.mExogenousInfo is not None):
            df = self.mExogenousInfo.transformDataset(df);
        lag_df = self.generateLagsForForecast(df);
        inputs = lag_df[self.mInputNames].values
        if(self.is_autoregressive_model()):
            inputs = df[series].values
        pred = self.predict(inputs)
        df[self.mOutName] = pred[pred.columns[1]];            
        self.compute_ar_residue(df)
        return df;

class cCaret_Model(cR_Model):
    def __init__(self , cycle_residue_name, P , iExogenousInfo = None):
        super().__init__(cycle_residue_name, P, iExogenousInfo)
        self.mMethod = "earth";
        self.mCommonModelName = "MARS";
        self.mCaretParams = None

    def set_name(self):
        lCommonName = self.mCommonModelName + ("" if(self.mExogenousInfo is None) else "X")
        self.mOutName = self.mCycleResidueName +  '_' + lCommonName + '(' + str(self.mNbLags) + ")";
        self.mFormula = self.mCommonModelName
        lPrefix = self.generate_temp_name()
        self.mModelName = self.mDirName + "/caret_" + self.get_method_as_string() + "_" + lPrefix;        
        
    def dumpCoefficients(self, iMax=10):
        logger = tsutil.get_pyaf_logger();
        logger.info("PYAF_USING_CARET_METHOD " + str((self.mCommonModelName, self.mMethod)));
        
    def get_method_as_string(self):
        return str(self.mMethod)

    def add_specific_train_code(self):
        lScript = self.add_needed_library("caret")
        lScript = lScript + 'model = train( TGT ~ ., data=df, method="' +self.mMethod + '");\n'
        return lScript

    def add_specific_predict_code(self, X):
        pred_type = "raw"
        lScript = self.add_needed_library("caret")
        lScript = lScript + 'predicted = predict(reloaded_model, newdata=df, type=\"' + pred_type + '");\n'
        return lScript


class cTAR_Model(cR_Model):
    def __init__(self , cycle_residue_name, P , iExogenousInfo = None):
        super().__init__(cycle_residue_name, P, iExogenousInfo)
        self.mCommonModelName = "TAR";

    def is_autoregressive_model(self):
        return True
        
    def set_name(self):
        lCommonName = self.mCommonModelName + ("" if(self.mExogenousInfo is None) else "X")
        self.mOutName = self.mCycleResidueName +  '_' + lCommonName + '(' + str(self.mNbLags) + ")";
        self.mFormula = self.mCommonModelName
        lPrefix = self.generate_temp_name()
        self.mModelName = self.mDirName + "/threshold_ar_" + lPrefix;        
        
    def dumpCoefficients(self, iMax=10):
        logger = tsutil.get_pyaf_logger();
        logger.info("PYAF_USING_NTS_TAR_MODEL " + str(self.mCommonModelName))
        
    def add_specific_train_code(self):
        lScript = self.add_needed_library("NTS")
        lScript = lScript + 'thresholds.est = uTAR(y=df$TGT, p1=2, p2=2, d=2, thrQ=c(0,1), Trim=c(0.1,0.9), include.mean=TRUE, method="NeSS", k0=50);\n'
        lScript = lScript + 'model = uTAR.est(y=df$TGT, , arorder=c(2,2), thr=thresholds.est$thr, d=2);\n'
        return lScript

    def add_specific_predict_code(self, X):
        lScript = self.add_needed_library("NTS")
        lScript = lScript + 'predicted = uTAR.pred(mode=reloaded_model, orig=0 , h=' + str(X.shape[0]) + ' - sum(reloaded_model$nobs),iterations=100,ci=0.95,output=TRUE)\n'
        lScript = lScript + "nempty = length(reloaded_model$data) -  length(reloaded_model$residuals)\n" # empty residues at the beginning (AR)
        lScript = lScript + 'residuals = rbind(matrix(0, nempty) , matrix(reloaded_model$residuals))\n'
        lScript = lScript + 'data = reloaded_model$data\n'
        lScript = lScript + 'fitted = data + residuals\n'
        lScript = lScript + 'predicted = rbind(fitted, predicted$pred)\n'
        return lScript

class cMARS_Model(cCaret_Model):
    def __init__(self , cycle_residue_name, P , iExogenousInfo = None):
        super().__init__(cycle_residue_name, P, iExogenousInfo)
        self.mMethod = "earth"
        self.mCommonModelName = "MARS"
