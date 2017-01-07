import WS.WS_Backend as be

# this is a copy-paste from the API post data
lDict = {
    "CSVFile": "http://chart.finance.yahoo.com/table.csv?s=GOOG&a=8&b=14&c=2015&d=9&e=14&f=2016&g=d&ignore=.csv", 
    "DateFormat": "%Y-%m-%d", 
    "Horizon": 21, 
    "Name": "PYAF_MODEL_DJOQFD", 
    "Present": "2016-01-01", 
    "SignalVar": "Close", 
    "TimeVar": "Date"
    }

lModel = be.cWSModel();
lModel.from_dict(lDict);

print("\n\n<ModelInfo>")
print(lModel.mForecastEngine.to_json());
print("</ModelInfo>\n\n")
