import numpy as np
import pandas as pd
import pyaf.ForecastEngine as autof


df = pd.DataFrame([[0 , 0.54543]], columns = ['date' , 'signal'])
lEngine = autof.cForecastEngine()
lEngine.train(df , 'date' , 'signal', 1);

