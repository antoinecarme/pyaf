import numpy as np
import pandas as pd
import datetime

csvfile_link = "data/ozone-la-exogenous-2.csv"
exog_dataframe = pd.read_csv(csvfile_link);
exog_dataframe['Date'] = exog_dataframe['Date'].astype(np.datetime64);
# print(exog_dataframe.info())
exog_dataframe.head()

import pyaf.ForecastEngine as autof
lEngine = autof.cForecastEngine()

csvfile_link = "https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/ozone-la.csv"
ozone_dataframe = pd.read_csv(csvfile_link);
ozone_dataframe['Date'] = ozone_dataframe['Month'].apply(lambda x : datetime.datetime.strptime(x, "%Y-%m"))

# print(ozone_dataframe.info())

lExogenousData = (exog_dataframe , ['Exog2' , 'Exog3' , 'Exog4',  'Exog5']) 

lEngine.train(ozone_dataframe , 'Date' , 'Ozone', 12 , lExogenousData);
lEngine.getModelInfo();
print(lEngine.mSignalDecomposition.mTrPerfDetails.head());

forecast_dataframe = lEngine.forecast(ozone_dataframe, 12)
print(forecast_dataframe.tail(24))
