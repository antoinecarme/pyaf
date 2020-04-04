import tests.heroku.build_generic_heroku_model as builder

# this is a copy-paste from the API post data
lDict = {
        "CSVFile": "https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/ozone-la.csv",
        "DateFormat": "%Y-%m", 
        "Horizon": 12, 
        "Name": "model1", 
        "Present": "1968-08", 
        "SignalVar": "Ozone", 
        "TimeVar": "Month"
      }

builder.build_model(lDict)
