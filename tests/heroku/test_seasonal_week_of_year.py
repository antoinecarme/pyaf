import tests.heroku.build_generic_heroku_model as builder

# this is a copy-paste from the API post data
# swithchinfg to github data source .
# Unfortunately, yahoo stopped its excellent financial data service ...

lDict = {
    "CSVFile": "https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/a5fa96af431caabc817180d52bc1d01b8c61da41/YahooFinance/nasdaq/yahoo_GOOG.csv",
    "DateFormat": "%Y-%m-%d",
    "Horizon": 21, 
    "Name": "PYAF_MODEL_DJOQFD_GOOG", 
    "Present": "2016-01-01", 
    "SignalVar": "Close", 
    "TimeVar": "Date"
    }

builder.build_model(lDict)
