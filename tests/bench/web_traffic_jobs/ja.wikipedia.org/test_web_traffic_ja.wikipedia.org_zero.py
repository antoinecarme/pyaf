import pyaf.Bench.web_traffic.Forecaster as fo


PROJECTS = ['ja.wikipedia.org']

data_dir = 'data/web-traffic-time-series-forecasting'

lForecaster = fo.cProjectForecaster()
lForecaster.mDataDirectory = data_dir
lForecaster.mBackendName = 'zero'
lForecaster.mKeysFileName = 'key_1.csv.zip'
last_date = '2016-12-31'
horizon = 60
lForecaster.mKeysFileName = 'key_1.csv.zip'
lForecaster.forecast(PROJECTS, last_date , horizon)
