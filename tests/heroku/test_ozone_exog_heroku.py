import tests.heroku.build_generic_heroku_model as builder

# this is a copy-paste from the API post data
lDict = {
    "CSVFile": "https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/ozone-la.csv",
    "DateFormat": "%Y-%m", 
    "Horizon": 12, 
    "Name": "model_ozone_exog", 
    "Present": "", 
    "SignalVar": "Ozone", 
    "TimeVar": "Month",
    "ExogenousData":"https://raw.githubusercontent.com/antoinecarme/pyaf/master/data/ozone-la-exogenous-3.csv",
    # "ExogenousData":"data/ozone-la-exogenous-3.csv",
    }

print(sorted(lDict.items()))
builder.build_model(lDict)
