import pandas as pd
import numpy as np
import pyaf.HierarchicalForecastEngine as hautof

import datetime

#get_ipython().magic('matplotlib inline')

import pandas as pd

filename = "https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/Stat_FR_CommExt/merged/French_Wine_Export_in_Euros_Some_Countries.csv"

French_Wine_Export_in_Euros_DF = pd.read_csv(filename);
lDateColumn = 'Month';
French_Wine_Export_in_Euros_DF[lDateColumn] = French_Wine_Export_in_Euros_DF[lDateColumn].apply(lambda x : datetime.datetime.strptime(x, "%Y-%m-%d"))

French_Wine_Export_in_Euros_DF.head(5)


Wines = u"ALSACE BEAUJOLAIS BORDEAUX BOURGOGNE CHAMPAGNE EUROPE FRANCE LANGUEDOC LOIRE OTHER RHONE".split();
Wines = Wines[0:2];
Variants = ['BLANC' , 'MOUSSEUX' , 'ROUGE'];
Variants = ['BLANC' , 'ROUGE'];
Countries = ['GB', 'US', 'DE', 'BE', 'CN', 'JP', 'CH', 'HK', 'NL', 'CA' , 'OTHER']
Regions = ['EUROPE', 'AMERICA', 'EUROPE' , 'EUROPE' , 'ASIA' , 'ASIA' , 'EUROPE',  'ASIA', 'EUROPE' , 'AMERICA' , 'OTHER_REGION']
lDict = dict(zip(Countries , Regions));

Countries = Countries[0:6]

rows_list = [];
for v in Variants:
    for w in Wines:
        for c in Countries:
            col = w + "_" + v + "_" + c;
            region = lDict[c]
            if(col in French_Wine_Export_in_Euros_DF.columns):
                rows_list.append([col , c , region , 'WORLD']);
            
lLevels = ['Wine' , 'Country' , 'Region' , 'WORLD'];
lHierarchy = {};
lHierarchy['Levels'] = lLevels;
lHierarchy['Data'] = pd.DataFrame(rows_list, columns =  lLevels);
lHierarchy['Type'] = "Hierarchical";
    
print(lHierarchy['Data'].head(lHierarchy['Data'].shape[0]));

lEngine = hautof.cHierarchicalForecastEngine()
lEngine.mOptions.mHierarchicalCombinationMethod = "BU";
lEngine

N = French_Wine_Export_in_Euros_DF.shape[0];
train_df = French_Wine_Export_in_Euros_DF.head(N-4);
H = 4;
lSignalVar = "Sales";

# lEngine.mOptions.enable_slow_mode();
# lEngine.mOptions.mDebugPerformance = True;
lEngine.train(train_df , lDateColumn , lSignalVar, H, lHierarchy, None);

lEngine.getModelInfo();
lEngine.standardPlots("outputs/WineExport");

dfapp_in = train_df.copy();
dfapp_in.tail()

dfapp_out = lEngine.forecast(dfapp_in, H);
dfapp_out.to_csv("outputs/Hierarchical_WineExport_apply_out.csv")
