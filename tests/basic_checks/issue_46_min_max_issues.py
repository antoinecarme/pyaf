import numpy as np
import pandas as pd

df = pd.DataFrame([[0 , 0.54543]], columns = ['date' , 'signal'])
# all dates are int64 , min is also int64 and max is int64
df['date'] = df['date'].astype(np.int64)
df.info()
print("min_max_date" , pd.Series([df['date'].min()]).dtype,
      pd.Series([df['date'].max()]).dtype)

# all dates are int64 , min is also int64 and max is int64
df['date'] = df['date'].astype(np.int32)
df.info()
print("min_max_date" , pd.Series([df['date'].min()]).dtype,
      pd.Series([df['date'].max()]).dtype)

# now use, pandas series
