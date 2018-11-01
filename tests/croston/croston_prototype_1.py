# source : https://otexts.org/fpp2/counts.html

import pandas as pd

# %matplotlib inline

lCounts = "0 2 0 1 0 11 0 0 0 0 2 0 6 3 0 0 0 0 0 7 0 0 0 0 0 0 0 3 1 0 0 1 0 1 0 0".split()
lCounts = [float(c) for c in lCounts]
N = len(lCounts)
lDates = pd.date_range(start="2000-01-01", periods=N, freq='m')

df = pd.DataFrame({"Date" : lDates, "Count" : lCounts})

#  q is often called the “demand” and a the “inter-arrival time”.
q = df[abs(df['Count']) > 0.0]['Count']
demand_times = pd.Series(list(q.index)) + 1
a = demand_times - demand_times.shift(1).fillna(0.0)
df2 = pd.DataFrame({'demand_time' : list(demand_times), 'q' : list(q) , 'a' : list(a) })
df2

def get_coeff(alpha , croston_type):
    if(croston_type == "sba"):
        return 1.0-(alpha/2.0)
    elif(croston_type == "sbj"):
        return (1.0 - alpha/(2.0-alpha))
    # default 
    return 1.0



# q  and a forecast
alpha = 0.1

df2['q_est'] = None
df2['a_est'] = None

df2.loc[0 , 'q_est'] = df2['q'][0]
df2.loc[0,  'a_est'] = df2['a'][0]
for i in range(df2.shape[0] - 1):
    q1 = (1.0 - alpha) * df2['q_est'][ i ] + alpha * df2['q'][ i ]
    a1 = (1.0 - alpha) * df2['a_est'][ i ] + alpha * df2['a'][ i ]
    df2.loc[i + 1, 'q_est'] = q1
    df2.loc[i + 1, 'a_est'] = a1

coeff = get_coeff(alpha , "default")
df2['forecast'] = coeff * df2['q_est'] / df2['a_est']
df2



forecast_11 = df2['q_est'][10] / df2['a_est'][10]
forecast_11


df2['index'] = df2['demand_time'] - 1


df1 = df.reset_index()
df3 = df1.merge(df2 , how='left', on=('index' , 'index'))

df4 = df3.fillna(method='ffill')

print(df4)

# df4.plot('Date', ['Count' , 'forecast'])
