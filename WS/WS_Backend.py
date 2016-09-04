import pandas as pd
import numpy as np
import datetime as dt

import AutoForecast as autof
from Bench import TS_datasets as tsds

from flask import Flask, jsonify, request

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

    def convert_string_to_date(self, iString):
        if(self.mDateFormat is not None and self.mDateFormat != ""):
            return dt.datetime.strptime(iString, self.mDateFormat);
        return float(iString)

    def guess_Columns_if_needed(self):
        print("DATASET_DETECTED_COLUMNS", self.mFullDataFrame.columns);
        if(len(self.mFullDataFrame.columns) == 1):
            lDataFrame = pd.DataFrame();
            lDataFrame[self.mTimeVar] = range(0, self.mFullDataFrame.shape[0]);
            lLastColumn = self.mFullDataFrame.columns[-1];
            lDataFrame[self.mSignalVar] = self.mFullDataFrame[lLastColumn]
            self.mFullDataFrame = lDataFrame;
        else:
            if(self.mTimeVar not in self.mFullDataFrame.columns):
                lFirstColumn = self.mFullDataFrame.columns[0];
                self.mFullDataFrame[self.mTimeVar] = self.mFullDataFrame[lFirstColumn]
            if(self.mSignalVar not in self.mFullDataFrame.columns):
                lLastColumn = self.mFullDataFrame.columns[-1];
                self.mFullDataFrame[self.mSignalVar] = self.mFullDataFrame[lLastColumn]
        print("DATASET_FINAL_COLUMNS", self.mFullDataFrame.columns);
                

    def readData(self):
        self.mFullDataFrame = pd.read_csv(self.mCSVFile, sep=r',', engine='python');
        self.guess_Columns_if_needed();
        self.mFullDataFrame[self.mTimeVar] = self.mFullDataFrame[self.mTimeVar].apply(self.convert_string_to_date);
        self.mPresent = self.convert_string_to_date(self.mPresentTime);
        
    def trainModel(self):
        self.mTrainDataFrame = self.mFullDataFrame[self.mFullDataFrame[self.mTimeVar] <= self.mPresent];
        self.mAutoForecast = autof.cAutoForecast()
        self.mAutoForecast.train(self.mTrainDataFrame , self.mTimeVar , self.mSignalVar, self.mHorizon);        

    def applyModel(self):
        self.mApplyIn = self.mTrainDataFrame;
        self.mDetailedForecast_DataFrame = self.mAutoForecast.forecast(self.mApplyIn, self.mHorizon);
        self.mForecast_DataFrame = self.mDetailedForecast_DataFrame[[self.mTimeVar , self.mSignalVar, self.mSignalVar + '_BestModelForecast']]
        self.mForecastData = self.mForecast_DataFrame.tail(self.mHorizon);

    def create(self):
        self.readData();
        self.trainModel();
        self.applyModel();

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
        

    def as_dict(self):
        lForecastData = None;
        if(self.mForecastData is not None):
            lForecastData = self.mForecastData.to_json(date_format='iso');
        lModelInfo = self.getModelInfo();
        obj_d = {
            "Name" : self.mName,
            "CreationDate" : self.mCreationDate.isoformat(),            
            'CSVFile': self.mCSVFile,
            'DateFormat': self.mDateFormat,
            "SignalVar" : self.mSignalVar,
            "TimeVar" : self.mTimeVar,
            "Present" : self.mPresentTime,
            "Horizon" : self.mHorizon,
            "ModelInfo" : [lModelInfo],
            "ForecastData" : [lForecastData]
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

    # samples
    def fillSampleModels(self):
        lOzone = { "CSVFile" : "https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/ozone-la.csv", 
                   "DateFormat" : "%Y-%m", 
                   "SignalVar" : "Ozone",
                   "TimeVar" : "Month",
                   "Present" : "1960-08",
                   "Horizon" : 12,
                   "Name" : "Ozone_Model_12"
                   };
        
        lAirline = { "CSVFile" : "https://raw.githubusercontent.com/hawk31/nnet-ts/master/nnet_ts/AirPassengers.csv",
                     "DateFormat" : "", # not a date ... a number
                     "SignalVar" : "AirPassengers",
                     "TimeVar" : "Time",
                     "Present" : "100",
                     "Horizon" : 7,
                     "Name" : "AirPassengers_Model"
                     };
        self.add_model(lOzone);
        self.add_model(lAirline);
        
