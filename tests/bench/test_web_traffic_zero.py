import pyaf.Bench.web_traffic.Forecaster as fo

data_dir = "data/web-traffic-time-series-forecasting"

PROJECTS = ['commons.wikimedia.org', 'de.wikipedia.org',
            'es.wikipedia.org', 'www.mediawiki.org', 
            'fr.wikipedia.org', 'ru.wikipedia.org',
            'en.wikipedia.org', 'zh.wikipedia.org',
            'ja.wikipedia.org']


lForecaster = fo.cProjectForecaster()
lForecaster.mDataDirectory = data_dir
lForecaster.mKeysFileName = "key_1.csv.zip"

lForecaster.forecast(PROJECTS, '2016-12-31' , 60)

