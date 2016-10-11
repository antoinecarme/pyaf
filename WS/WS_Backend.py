import pandas as pd
import numpy as np
import datetime as dt

import AutoForecast.ForecastEngine as autof
import AutoForecast.Bench.TS_datasets as tsds

from flask import Flask, jsonify, request
import logging

# for timing
import time

class cWSModel:

    def __init__(self , name):
        self.mName = name;
        self.mCreationDate = dt.datetime.now();
        self.mSignalVar = None;
        self.mTimeVar = None;
        self.mPresentTime = None ; # "1960-08-01"; # start forecasting after this date.
        self.mHorizon = 1;
        self.mMaxHistoryForDisplay = 1000; # max length of data frames returned in json
        self.mForecastData = None;
        self.mPlots = {};
        self.mURI = "http://192.168.88.88:8080/";
        
    def convert_string_to_date(self, iString):
        if(self.mDateFormat is not None and self.mDateFormat != ""):
            return dt.datetime.strptime(iString, self.mDateFormat);
        return float(iString)

    def guess_Columns_if_needed(self):
        print("DATASET_DETECTED_COLUMNS", self.mFullDataFrame.columns);
        if(len(self.mFullDataFrame.columns) == 1):
            lLastColumn = self.mFullDataFrame.columns[-1];
            lDataFrame = pd.DataFrame();
            lDataFrame[self.mTimeVar] = range(0, self.mFullDataFrame.shape[0]);
            lDataFrame[self.mSignalVar] = self.mFullDataFrame[lLastColumn]
            self.mFullDataFrame = lDataFrame;
        else:
            if(self.mTimeVar not in self.mFullDataFrame.columns):
                lFirstColumn = self.mFullDataFrame.columns[0];
                self.mFullDataFrame[self.mTimeVar] = self.mFullDataFrame[lFirstColumn]
            if(self.mSignalVar not in self.mFullDataFrame.columns):
                lLastColumn = self.mFullDataFrame.columns[1];
                self.mFullDataFrame[self.mSignalVar] = self.mFullDataFrame[lLastColumn]
        print("DATASET_FINAL_COLUMNS", self.mFullDataFrame.columns);
                

    def readData(self):
        self.mFullDataFrame = pd.read_csv(self.mCSVFile, sep=r',', engine='python');
        self.guess_Columns_if_needed();
        self.mFullDataFrame[self.mTimeVar] = self.mFullDataFrame[self.mTimeVar].apply(self.convert_string_to_date);
        self.mFullDataFrame.sort_values(by = self.mTimeVar, inplace = True);
        self.mPresent = self.convert_string_to_date(self.mPresentTime);
        
    def trainModel(self):
        self.mTrainDataFrame = self.mFullDataFrame[self.mFullDataFrame[self.mTimeVar] <= self.mPresent];
        self.mAutoForecast = autof.cForecastEngine()
        self.mAutoForecast.train(self.mTrainDataFrame , self.mTimeVar , self.mSignalVar, self.mHorizon);        

    def applyModel(self):
        self.mApplyIn = self.mTrainDataFrame;
        self.mDetailedForecast_DataFrame = self.mAutoForecast.forecast(self.mApplyIn, self.mHorizon);
        self.mForecast_DataFrame = self.mDetailedForecast_DataFrame; # [[self.mTimeVar , self.mSignalVar, self.mSignalVar + '_Forecast']];
        self.mForecastData = self.mForecast_DataFrame.tail(self.mHorizon);
        lForecastName = self.mSignalVar + '_Forecast';
        self.mForecast = self.mForecastData[[self.mTimeVar, lForecastName,
                                             lForecastName + "_Lower_Bound",
                                             lForecastName + "_Upper_Bound"]]

    def generateCode(self):
        logger = logging.getLogger(__name__)
        self.mSQL = {};
        try:
            self.mSQL["Default"] = self.mAutoForecast.generateCode(iDSN = None, iDialect = None);
            self.mSQL["postgresql"] = self.mAutoForecast.generateCode(iDSN = None, iDialect = "postgresql");
            self.mSQL["mssql"] = self.mAutoForecast.generateCode(iDSN = None, iDialect = "mssql");
            self.mSQL["oracle"] = self.mAutoForecast.generateCode(iDSN = None, iDialect = "oracle");
            self.mSQL["mysql"] = self.mAutoForecast.generateCode(iDSN = None, iDialect = "mysql");
            self.mSQL["sybase"] = self.mAutoForecast.generateCode(iDSN = None, iDialect = "sybase");
            self.mSQL["sqlite"] = self.mAutoForecast.generateCode(iDSN = None, iDialect = "sqlite");
        except Exception as e:
            # logger.error("FAILED_TO_GENERATE_CODE_FOR " + self.mName + " " + str(e));
            raise
            pass

    def generatePlots(self):
        logger = logging.getLogger(__name__)
        self.mPlots = {};
        try:
            self.mPlots = self.mAutoForecast.getPlotsAsDict();
        except Exception as e:
            # logger.error("FAILED_TO_GENERATE_PLOTS " + self.mName + " " + str(e));
            # raise
            pass

    def create(self):
        start_time = time.time()
        self.readData();
        self.trainModel();
        self.applyModel();
        self.generateCode();
        self.generatePlots();
        self.mTrainingTime = time.time() - start_time;

    def update(self):
        self.create();

    def getModelInfo(self):
        return(self.mAutoForecast.to_json());

    def getForecasts(self):
        return self.mForecastData.to_json(date_format='iso');

    def from_dict(self, json_dict):
        self.mCSVFile = json_dict['CSVFile'];
        self.mDateFormat = json_dict['DateFormat'];
        self.mSignalVar = json_dict['SignalVar'];      
        self.mTimeVar = json_dict['TimeVar'];      
        self.mPresentTime = json_dict['Present'];      
        self.mHorizon = int(json_dict['Horizon']);      
        self.create();
        

    def get_dataset_info(self):
        lDatasetInfo = {};
        lDatasetInfo["Signal_Stats"] = {"Length" : str(self.mFullDataFrame[self.mSignalVar].shape[0]),
                                        "Min" : str(self.mFullDataFrame[self.mSignalVar].min()),
                                        "Max" : str(self.mFullDataFrame[self.mSignalVar].max()),
                                        "Mean" : str(self.mFullDataFrame[self.mSignalVar].mean()),
                                        "StdDev" : str(self.mFullDataFrame[self.mSignalVar].std()),
                                        };
        lDatasetInfo["Time_Stats"] = {"Min" : str(self.mFullDataFrame[self.mTimeVar].min()),
                                      "Max" : str(self.mFullDataFrame[self.mTimeVar].max()),
                                      };
        return lDatasetInfo;

    def as_dict(self):
        lForecastName = self.mSignalVar + '_Forecast';
        lForecastData = {};
        if(self.mForecastData is not None):
            lForecastData["Time"] = self.mForecast[self.mTimeVar].apply(str).tolist();
            lForecastData["Forecast"] = self.mForecast[lForecastName].tolist();
            lForecastData["Forecast" + "_Lower_Bound"] = self.mForecast[lForecastName + "_Lower_Bound"].tolist();
            lForecastData["Forecast" + "_Upper_Bound"] = self.mForecast[lForecastName + "_Upper_Bound"].tolist();

        lModelInfo = self.getModelInfo();
        lTrainOptions =  {
            'CSVFile': self.mCSVFile,
            'DateFormat': self.mDateFormat,
            "SignalVar" : self.mSignalVar,
            "TimeVar" : self.mTimeVar,
            "Present" : self.mPresentTime,
            "Horizon" : self.mHorizon,
        }
        lPlotLinks = {};
        lPlotLinks["all"] = self.mURI + "model/" + self.mName + "/plot/" + "all";
        for k in self.mPlots.keys():
            lPlotLinks[k] = self.mURI + "model/" + self.mName + "/plot/" + k;
        lTrainOptionsDescription =  {
            'CSVFile': "A CSV file (URIs are also welcome!!!) containing a date column (optional, a integer sequence starting at zero is used if not present), and a signal column, for which the future values are to be predicted. ",
            'DateFormat': "The format of the date column , if it is a physcial date/time/datetime column (iso : yyyy-mm-dd by default), empty otherwise",
            "SignalVar" : "Name of the signal column to be predicted",
            "TimeVar" : "Name of the date/time column",
            "Present" : "date/time of the last known signal value. Predictions start after this date/time",
            "Horizon" : "number of future time periods to be predicted. The length of a period is inferred from data (most frequent difference between two consecutive dates)",
            "Name" : "Name used to identify the model in the API"
        }

        lSQLLinks = {};
        for k in self.mSQL.keys():
            lSQLLinks[k] = self.mURI + "model/" + self.mName + "/SQL/" + k;

        lMetaData = {
            "Name" : self.mName,
            "ModelFormat" : "0.1",
            "CreationDate" : str(self.mCreationDate),
            "Training_Time" : str(self.mTrainingTime)
            }
        
        obj_d = {
            'MetaData' : [ lMetaData ],
            'TrainOptions': [ lTrainOptions ],
            'TrainOptionsHelp': [ lTrainOptionsDescription ],
            "CSVFileInfo" : [ self.get_dataset_info() ],
            "ModelInfo" : [lModelInfo],
            "ForecastData" : [lForecastData],
            "SQL" : [ lSQLLinks ],
            "Plots" : [ lPlotLinks ]
        }
        return obj_d

    @property
    def json(self):
        return jsonify(**self.as_dict())


class cFlaskBackend:

    def __init__(self):
        self.mName = None;
        self.models = {};
        self.fillSampleModels();

    # Models
    def get_model(self, name):
        model  = self.models.get(name);
        return model;


    def add_model(self, json_dict):
        name = json_dict['Name'];
        model  = cWSModel(name);
        model.from_dict(json_dict);
        self.models[name] = model;
        pass

    def update_model(self, name, value):
        model  = self.models.get(name);
        if(model):
            model.update(value);
        pass

    def remove_model(self, name):
        model  = self.models.get(name);
        if(model):
            del self.models[name];
        pass


    def add_yahoo_symbol(self, symbol):
        lSymbol = symbol
        lYahooURL = "http://ichart.finance.yahoo.com/table.csv?g=d&f=2015&e=12&c=2014&b=10&a=7&d=7&s=" + lSymbol;
        lYahoo = { "CSVFile" : lYahooURL,
                   "DateFormat" : "%Y-%m-%d", # ISO format
                   "SignalVar" : "Open",
                   "TimeVar" : "Date",
                   "Present" : "2014-12-31",
                   "Horizon" : 7,
                   "Name" : lSymbol + "_Model"
                   };
        self.add_model(lYahoo);
        

    # samples
    def fillSampleModels(self):
        lOzone = { "CSVFile" : "https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/ozone-la.csv", 
                   "DateFormat" : "%Y-%m", 
                   "SignalVar" : "Ozone",
                   "TimeVar" : "Month",
                   "Present" : "1968-08",
                   "Horizon" : 12,
                   "Name" : "Ozone_Model_12"
                   };
        
        lAirline = { "CSVFile" : "https://raw.githubusercontent.com/hawk31/nnet-ts/master/nnet_ts/AirPassengers.csv",
                     "DateFormat" : "", # not a date ... a number
                     "SignalVar" : "AirPassengers",
                     "TimeVar" : "Time",
                     "Present" : "150",
                     "Horizon" : 7,
                     "Name" : "AirPassengers_Model"
                     };
        self.add_model(lOzone);
        self.add_model(lAirline);
        # yahoo symbols ... online
        self.add_yahoo_symbol("AAPL")
        self.add_yahoo_symbol("GOOG")
        self.add_yahoo_symbol("MSFT")
        self.add_yahoo_symbol("SAP")
        self.add_yahoo_symbol("^FCHI")
