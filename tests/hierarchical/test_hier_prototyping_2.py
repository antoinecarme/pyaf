import pandas as pd
import numpy as np
import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds

import datetime

#get_ipython().magic('matplotlib inline')


def read_dataset():
    trainfile = "data/Hierarchical/hts_dataset.csv"
    lDateColumn = 'Date'

    df = pd.read_csv(trainfile, sep=r',', engine='python', skiprows=0);
    df[lDateColumn] = df[lDateColumn].apply(lambda x : datetime.datetime.strptime(x, "%Y-%m-%d"))

    print(df.tail(10))
    # df[:-10].tail()
    # df[:-10:-1]
    print(df.info())
    print(df.describe())
    return df;


def define_hierarchy_info():
    rows_list = [];
    # Sydney    NSW  Melbourne    VIC  BrisbaneGC    QLD  Capitals Other
    rows_list.append(['Sydney' , 'NSW_State' , 'Australia']);
    rows_list.append(['NSW' , 'NSW_State' , 'Australia']);
    # rows_list.append(['Melbourne' , 'VIC_State' , 'Australia']);
    # rows_list.append(['VIC' , 'VIC_State' , 'Australia']);
    # rows_list.append(['BrisbaneGC' , 'QLD_State' , 'Australia']);
    # rows_list.append(['QLD' , 'QLD_State' , 'Australia']);
    rows_list.append(['Capitals' , 'Other_State' , 'Australia']);
    rows_list.append(['Other' , 'Other_State' , 'Australia']);

    lLevels = ['City' , 'State' , 'Country'];
    lHierarcyInfo = {};
    lHierarcyInfo['Levels'] = lLevels;
    lHierarcyInfo['Data'] = pd.DataFrame(rows_list, columns =  lLevels);
    
    print(lHierarcyInfo['Data'].head(lHierarcyInfo['Data'].shape[0]));
    return lHierarcyInfo;

def enrich_dataset(df ,  hier):
    df1 = df.copy();
    lLevelCount = len(hier['Levels']);
    lContent = {};
    df = hier['Data'];
    for level in range(lLevelCount):
        lContent[level] = {};
    for row in range(df.shape[0]):
        for level in range(lLevelCount):
            col = df[df.columns[level]][row];
            if(col not in lContent[level].keys()):
                lContent[level][col] = set();
            if(level > 0):
                col1 = df[df.columns[level - 1]][row];
                lContent[level][col].add(col1);    
    print(lContent);
    for level in range(lLevelCount):
        if(level > 0):
            for col in lContent[level].keys():
                new_col = None;
                for col1 in lContent[level][col]:
                    if(new_col is None):
                        new_col = df1[col1];
                    else:
                        new_col = new_col + df1[col1];
                df1[col] = new_col;
    return df1;
        

df = read_dataset();
hier = define_hierarchy_info();
df1 = enrich_dataset(df, hier);

print(df1.head())

lDateColumn = 'Date'
lAllLevelColumns = [col for col in df1.columns if col != lDateColumn]
print("ALL_LEVEL_COLUMNS" , lAllLevelColumns);

H = 4;

for signal in lAllLevelColumns:
    lEngine = autof.cForecastEngine()
    lEngine

    # lEngine.mOptions.enable_slow_mode();
    # lEngine.mOptions.mDebugPerformance = True;
    lEngine.mOptions.set_active_autoregressions([]);

    lEngine.train(df1 , lDateColumn , signal, H);
    lEngine.getModelInfo();
    

    lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution

    #lEngine.standardPlots("outputs/hierarchical_" + signal);

    dfapp_in = df1.copy();
    dfapp_in.tail()

    dfapp_out = lEngine.forecast(dfapp_in, H);
    #dfapp_out.to_csv("outputs/ozone_apply_out.csv")
    dfapp_out.tail(2 * H)
    print("Forecast Columns " , dfapp_out.columns);
    Forecast_DF = dfapp_out[[lDateColumn , signal, signal + '_Forecast']]
    print(Forecast_DF.info())
    print("Forecasts\n" , Forecast_DF.tail(H));

    print("\n\n<ModelInfo>")
    print(lEngine.to_json());
    print("</ModelInfo>\n\n")
    print("\n\n<Forecast>")
    print(Forecast_DF.tail(2*H).to_json(date_format='iso'))
    print("</Forecast>\n\n")

