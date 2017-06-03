# PyAF web service sample usage

This web service is running on heroku at http://pyaf.herokuapp.com/

# Basic usage :

The job specification is transmitted in the JSON body of the request. The format is :

```
      {
        "CSVFile": "https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/ozone-la.csv", 
        "DateFormat": "%Y-%m", 
        "Horizon": 12, 
        "Present": "1968-08", 
        "SignalVar": "Ozone", 
        "TimeVar": "Month"
        "Name": "my_model_name"
      }
```

The result is a JSON format containing, the model info, the forecast values etc

# Plots

Once the model is built, A set of (6 ?) plots is available and the uris are given in the JSON result

        "AR": "https://pyaf.herokuapp.com/model/my_model_name/plot/AR", 
        "Cycle": "https://pyaf.herokuapp.com/model/my_model_name/plot/Cycle", 
        "Forecast": "https://pyaf.herokuapp.com/model/my_model_name/plot/Forecast", 
        "Prediction_Intervals": "https://pyaf.herokuapp.com/model/my_model_name/plot/Prediction_Intervals", 
        "Trend": "https://pyaf.herokuapp.com/model/my_model_name/plot/Trend", 
        "all": "https://pyaf.herokuapp.com/model/my_model_name/plot/all"

These are base-64 encoded jpg images.

# Curl command line

```
PYAF_URL="http://pyaf.herokuapp.com/model"
CONTENT_TYPE="Content-Type: application/json"
CSV_URI = "http://chart.finance.yahoo.com/table.csv?s=GOOG&a=8&b=14&c=2015&d=9&e=14&f=2016&g=d&ignore=.csv"
BODY_DATA="{"SignalVar":"Close", "Horizon":"21", "Present":"2016-01-01", "CSVFile": "$CSV_URI"}";

curl --header "$CONTENT_TYPE" -X POST --data "$BODY_DATA" "$PYAF_URL"

```

See [curl_tests.sh](curl_tests.sh) for more examples.

# Python client

See [test_heroku_web_service.py](test_heroku_web_service.py).
