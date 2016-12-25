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
    lStructure = {};
    df = hier['Data'];
    for level in range(lLevelCount):
        lStructure[level] = {};
    for row in range(df.shape[0]):
        for level in range(lLevelCount):
            col = df[df.columns[level]][row];
            if(col not in lStructure[level].keys()):
                lStructure[level][col] = set();
            if(level > 0):
                col1 = df[df.columns[level - 1]][row];
                lStructure[level][col].add(col1);    
    print(lStructure);
    for level in range(lLevelCount):
        if(level > 0):
            for col in lStructure[level].keys():
                new_col = None;
                for col1 in lStructure[level][col]:
                    if(new_col is None):
                        new_col = df1[col1];
                    else:
                        new_col = new_col + df1[col1];
                df1[col] = new_col;
    return (lStructure, df1);


def createModelForAllLevels(df, hier, iStrcture, H, iDateColumn):
    lModels = {};
    lLevelCount = len(hier['Levels']);
    for level in range(lLevelCount):
        lModels[level] = {};
        for signal in iStrcture[level].keys():
            lEngine = autof.cForecastEngine()
            lEngine

            # lEngine.mOptions.enable_slow_mode();
            # lEngine.mOptions.mDebugPerformance = True;
            lEngine.mOptions.set_active_autoregressions([]);
            lEngine.train(df1 , lDateColumn , signal, H);
            lEngine.getModelInfo();
            lModels[level][signal] = lEngine;
    return lModels;     

def forecastAllModels(df, hier, iStructure, iModels, H, iDateColumn):
    lForecast_DF = pd.DataFrame();
    lForecast_DF[iDateColumn] = df[iDateColumn]
    for level in iModels.keys():
        for signal in iModels[level].keys():
            lEngine = iModels[level][signal];
            dfapp_in = df.copy();
            # dfapp_in.tail()

            dfapp_out = lEngine.forecast(dfapp_in, H);
            print("Forecast Columns " , dfapp_out.columns);
            lForecast_DF[signal] = dfapp_out[signal]
            lForecast_DF[signal + '_Forecast'] = dfapp_out[signal + '_Forecast']
    print(lForecast_DF.columns);
    print(lForecast_DF.head());
    print(lForecast_DF.tail());
    return lForecast_DF;


def reportOnBottomUpForecasts(iStructure, iModels, iForecast_DF, iDateColumn):
    lForecast_DF_BU = pd.DataFrame();
    lForecast_DF_BU[iDateColumn] = iForecast_DF[iDateColumn];
    lPerfs = {};
    for level in iStructure.keys():
        for signal in iStructure[level].keys():
            lEngine = iModels[level][signal];
            new_BU_forecast = None;
            for col1 in lStructure[level][signal]:
                if(new_BU_forecast is None):
                    new_BU_forecast = iForecast_DF[col1 + "_Forecast"];
                else:
                    new_BU_forecast = new_BU_forecast + iForecast_DF[col1 + "_Forecast"];
            lForecast_DF_BU[signal] = iForecast_DF[signal];            
            lForecast_DF_BU[signal + "_Forecast"] = iForecast_DF[signal + "_Forecast"];
            if(new_BU_forecast is None):
                new_BU_forecast = iForecast_DF[signal + "_Forecast"];
            lForecast_DF_BU[signal + "_BU_Forecast"] = new_BU_forecast;
            lPerf = lEngine.computePerf(lForecast_DF_BU[signal], lForecast_DF_BU[signal + "_Forecast"], signal)
            lPerf_BU = lEngine.computePerf(lForecast_DF_BU[signal], lForecast_DF_BU[signal + "_BU_Forecast"],  signal + "_BU")
            lPerfs[signal] = (lPerf , lPerf_BU);
            
    print(lForecast_DF_BU.head());
    print(lForecast_DF_BU.tail());

    for (sig , perf) in lPerfs.items():
        print("PERF_REPORT_BU" , sig , perf[0].mL2,  perf[0].mMAPE, perf[1].mL2,  perf[1].mMAPE)
    return lForecast_DF_BU;
        
df = read_dataset();
hier = define_hierarchy_info();
(lStructure, df1) = enrich_dataset(df, hier);

print(df1.head())

lDateColumn = 'Date'
lAllLevelColumns = [col for col in df1.columns if col != lDateColumn]
print("ALL_LEVEL_COLUMNS" , lAllLevelColumns);

H = 4;

lModels = createModelForAllLevels(df1, hier, lStructure, H, lDateColumn);
lForecast_DF = forecastAllModels(df1, hier, lStructure, lModels, H, lDateColumn)
lForecast_DF_BU = reportOnBottomUpForecasts(lStructure, lModels, lForecast_DF, lDateColumn);

