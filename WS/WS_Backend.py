import pandas as pd
import numpy as np
import datetime as dt

import SignalDecomposition as SigDec
import TS_datasets as tsds

import sqlite3
from flask import Flask, jsonify, request

class cWSDataset:
    
    def __init__(self , name):
        self.mName = name;
        self.mCSVFile = None;
        self.mDateFormat = None;

    def update(self, value):
        self.mCSVFile = value;
        pass

    def as_dict(self):
        obj_d = {
            'Name' : self.mName,
            'CSVFile': self.mCSVFile,
            'DateFormat': self.mDateFormat
        }
        return obj_d


    def __html__(self):
        return json.dumps(**self.as_dict())
    
class cWSModel:

    def __init__(self , name):
        self.mName = name;
        self.mCreationDate = dt.datetime.now();
        self.mDataset = None;
        self.mSignalVar = "Ozone";
        self.mTimeVar = "Date";
        self.mHorizon = 1;
        self.mMaxHistoryForDisplay = 1000; # max length of data frames returned in json

    def readData(self):
        self.mDataFrame = pd.read_csv(self.mDataset.mCSVFile, sep=r',', engine='python', skiprows=1);
        self.mDataFrame['Time'] = self.mDataFrame['Month'].apply(lambda x : datetime.datetime.strptime(x, "%Y-%m"));        
    def trainModel(self):
        self.mDecomp = SigDec.cSignalDecomposition()
        self.mDecomp.train(self.mDataFrame , self.mTimeVar , self.mSignalVar, self.mHorizon);        

    def applyModel(self):
        self.mApplyIn = self.mDataFrame;
        self.mDetailedForecast_DataFrame = lDecomp.forecast(lApplyOut, self.mHorizon);
        self.mForecast_DataFrame = self.mDetailedForecast_DataFrame[[self.mTimeVar , self.mSignalVar, self.mSignalVar + '_BestModelForecast']]
        #print(Forecast_DF.info())
        self.mForecastData = self.mForecast_DataFrame.tail(H);

    def create(self):
        self.readData();
        self.trainModel();
        self.applyModel();

    def update(self):
        self.create();

    def getModelInfo(self):
        return(lDecomp.to_json());

    def getForecasts(self):
        return self.mForecastData.to_json(date_format='iso');

    def as_dict(self):
        obj_d = {
            "Name" : self.mName,
            "CreationDate" : self.mCreationDate.isoformat(),            
            "Dataset" : self.mDataset,
            "SignalVar" : self.mSignalVar,
            "TimeVar" : self.mTimeVar,
            "Horizon" : self.mHorizon
        }
        return obj_d

    @property
    def json(self):
        return jsonify(**self.as_dict())

class cFlaskBackend:

    def __init__(self):
        self.mName = None;
        self.models = {};        
        self.datasets = {};
        self.fillData();

    def fillData(self):
        self.datasets['Ozone'] = cWSDataset('Ozone');
        self.datasets['AirLine'] = cWSDataset('AirLine');
        
    def get_db(self):
        conn = sqlite3.connect(self.mName)

    # Datasets
    def get_dataset(self, name):
        ds  = self.datasets.get(name);
        return ds;

    def add_dataset(self, json_dict):
        name = json_dict['Name'];
        ds = cWSDataset(name);
        ds.mCSVFile = json_dict['CSVFile'];
        ds.mDateFormat = json_dict['DateFormat'];      
        self.datasets[name] = ds;
        return self.datasets;

    def update_dataset(self, name, value):
        ds  = self.datasets.get(name);
        if(ds):
            ds.update(value);
        return ds

    def remove_dataset(self, name):
        ds  = self.datasets.get(name);
        if(ds):
            del self.datasets[name];
        return self.datasets;


    # Models
    def get_model(self, name):
        model  = self.models.get(name);
        return model;


    def add_model(self, json_dict):
        name = json_dict['Name'];
        model  = cWSModel(name);
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
