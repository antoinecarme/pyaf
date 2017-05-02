import numpy as np
import pandas as pd
import pyaf.ForecastEngine as autof

try:
    df = pd.DataFrame([[0 , 0.54543]], columns = ['date' , 'signal'])
    lEngine = autof.cForecastEngine()
    lEngine.train(df , 'date_other' , 'signal', 1);
    raise Exception("NOT_OK")
except Exception as e:
    # should fail
    print(str(e));
    assert(str(e) == "PYAF_ERROR_TIME_COLUMN_NOT_FOUND date_other")
    if(str(e) == "NOT_OK"):
        raise
    pass
