import numpy as np
import pandas as pd
import pyaf.ForecastEngine as autof


df = pd.DataFrame([[0 , 0.54543]], columns = ['date' , 'signal'])

for ty in [np.int32 , np.float32, np.int64 , np.float64]:
    df['signal'] = df['signal'].astype(ty)
    lEngine = autof.cForecastEngine()
    lEngine.train(df , 'date' , 'signal', 1);

for ty in [object , bool]:
    try:
        df['signal'] = df['signal'].astype(ty)
        lEngine = autof.cForecastEngine()
        lEngine.train(df , 'date' , 'signal', 1);
        raise Exception("NOT_OK")
    except Exception as e:
        # should fail
        print(str(e));
        if(str(e) == "NOT_OK"):
            raise
        pass
    
for ty in [np.datetime64]:
    try:
        df['signal'] = df['signal'].apply(lambda x : np.datetime64('2005-02-25'))
        lEngine = autof.cForecastEngine()
        lEngine.train(df , 'date' , 'signal', 1);
        raise Exception("NOT_OK")
    except Exception as e:
        # should fail
        print(str(e));
        if(str(e) == "NOT_OK"):
            raise
        pass
    
