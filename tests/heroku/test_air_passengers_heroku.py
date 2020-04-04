import tests.heroku.build_generic_heroku_model as builder

# this is a copy-paste from the API post data
lDict = {
    "CSVFile": "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/AirPassengers.csv",
    "DateFormat": None, 
    "Horizon": 7, 
    "Name": "model_air_by_tests", 
    "Present": "", 
    "SignalVar": "AirPassengers",
    "TimeVar": "time"
    }

builder.build_model(lDict)
