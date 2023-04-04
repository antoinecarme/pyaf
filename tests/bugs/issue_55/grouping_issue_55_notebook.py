import matplotlib
matplotlib.use('Agg')
    
import pandas as pd

import datetime

filename = "https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/Stat_FR_CommExt/merged/French_Wine_Export_in_Euros_Some_Countries.csv"

French_Wine_Export_in_Euros_DF = pd.read_csv(filename);

lDateColumn = 'Month';
French_Wine_Export_in_Euros_DF[lDateColumn] = French_Wine_Export_in_Euros_DF[lDateColumn].apply(lambda x : datetime.datetime.strptime(str(x), "%Y-%m-%d"))

French_Wine_Export_in_Euros_DF.head(5)


French_Wine_Export_in_Euros_DF.describe()


CN_columns = [col for col in French_Wine_Export_in_Euros_DF.columns if col.endswith('_CN') ]
French_Wine_Export_in_Euros_DF[CN_columns].describe()


French_Wine_Export_in_Euros_DF.plot('Month' , 
                                    CN_columns,
                                   figsize=(32 , 12)).legend(fontsize=18)


Wines = "ALSACE BEAUJOLAIS BORDEAUX BOURGOGNE CHAMPAGNE EUROPE FRANCE LANGUEDOC LOIRE OTHER RHONE".split();
Variants = ['BLANC' , 'MOUSSEUX' , 'ROUGE'];
Countries = ['GB', 'US', 'DE', 'BE', 'CN', 'JP', 'CH', 'HK', 'NL', 'CA' , 'OTHER']

len(Wines) * len(Variants) * len(Countries) 


# from IPython.display import display


# In[13]:

rows = [];
cols = []
for cntry in Countries:
    cntry_columns = [col for col in French_Wine_Export_in_Euros_DF.columns if col.endswith('_' + cntry) ]
    cols = []
    row = [cntry];
    for col in cntry_columns:
        col1 = col.replace('_' + cntry , "")
        sum1 = French_Wine_Export_in_Euros_DF[col].sum()
        row.append(sum1);
        cols.append(col1);
    rows.append(row);

plot_df = pd.DataFrame(rows , columns=['Country'] + cols);

    
plot_df.set_index('Country').plot.barh(stacked=True, figsize=(20,10), fontsize = 16, colormap='Paired').legend(loc='best', fontsize=12)



# ## Grouping Definition

Regions = ['EUROPE', 'AMERICA', 'EUROPE' , 'EUROPE' , 'ASIA' , 'ASIA' , 'EUROPE',  'ASIA', 'EUROPE' , 'AMERICA' , 'OTHER_REGION']
lDict = dict(zip(Countries , Regions));

# simplify !!!!
Variants = ['BLANC' , 'ROUGE'];
Wines = Wines[0:3];
Countries = Countries[0:5]

lGroups = {}
lGroups["Country"] = Countries
lGroups["Variant"] = Variants
lGroups["Wine"] = Wines

lHierarchy = {};
lHierarchy['Levels'] = None;
lHierarchy['Data'] = None;
lHierarchy['Groups']= lGroups;
# the most important !!!!
lHierarchy['GroupOrder']= ["Wine", "Variant", "Country"]; # group by Wine first, then by variant, etc
lHierarchy['Type'] = "Grouped";
    

print(lHierarchy)


# create a model to plot the hierarchy.
import pyaf.HierarchicalForecastEngine as hautof
lEngine = hautof.cHierarchicalForecastEngine()


lSignalHierarchy = lEngine.plot_Hierarchy(French_Wine_Export_in_Euros_DF , "Month", "Signal", 1, 
                                          lHierarchy, None);


print(lSignalHierarchy.mStructure)


# create a hierarchical model and train it
import pyaf.HierarchicalForecastEngine as hautof

lEngine = hautof.cHierarchicalForecastEngine()

lSignalVar = "Sales";

#
N = French_Wine_Export_in_Euros_DF.shape[0];
H = 4;
train_df = French_Wine_Export_in_Euros_DF.head(N-H);


lSignalHierarchy = lEngine.train(train_df , lDateColumn, lSignalVar, 1, lHierarchy, None);

French_Wine_Export_in_Euros_DF.info()


lInfo = lEngine.to_dict()
print(lInfo.keys())

print(lInfo['Structure'])

print(lInfo['Models'].keys())

print(lInfo['Models']['BORDEAUX_ROUGE_CN'])

perfs = [];
for model in sorted(lInfo['Models'].keys()):
    lPerf = lInfo['Models'][model]['Model_Performance'][1]
    perfs.append([model , lPerf['RMSE'] , lPerf['MAPE']])
df_perf = pd.DataFrame(perfs , columns=['Model' , 'RMSE' , 'MAPE']);
df_perf = df_perf.sort_values(by = ['MAPE'])
print(df_perf)

lEngine.mSignalHierarchy.plot()

AllSignals_Engine = lEngine.mSignalHierarchy.mModels # __CN is at hierarchical level 2

AllSignals_Engine.getModelInfo()

CN_Model = AllSignals_Engine.mSignalDecomposition.mBestModels['__CN'] 
CN_Model.standardPlots("outputs/bugs_hier_grouping_issue_55_CN")

lEngine.mOptions.mHierarchicalCombinationMethod = ["BU" , 'TD' , 'MO' , 'OC'];
dfapp_out = lEngine.forecast(train_df, H);

def strip_leading_underscores_if_needed(col_name):
    # this is a workaround for some matplotlib bugs. legend cannot contain names starting with '_'
    patched_name = col_name[2:] if(col_name.startswith('__')) else col_name
    patched_name = patched_name[1:] if(patched_name.startswith('_')) else patched_name
    return patched_name

dfapp_out.columns = [strip_leading_underscores_if_needed(col) for col in dfapp_out.columns]

print(dfapp_out.columns)
dfapp_out.info()

for country in Countries:
    plot = dfapp_out.plot('Month' , 
                          [country , country + '_Forecast' , 
                           country + '_BU_Forecast',  
                           country + '_PHA_TD_Forecast',  
                           country + '_AHP_TD_Forecast'  ,  
                           country + '_MO_Forecast' ,
                           country + '_OC_Forecast'  ],
                          figsize=(32 , 12)).legend(fontsize=18)
    fig = plot.get_figure()
    fig.savefig("outputs/bugs_hier_grouping_issue_55_Forecast_" + country + ".png")

world = ''
plot = dfapp_out.plot('Month' , 
                      [world , world + 'Forecast' , 
                       world + 'BU_Forecast',  
                       world + 'PHA_TD_Forecast',  
                       world + 'AHP_TD_Forecast'  ,  
                       world + 'MO_Forecast' ,
                       world + 'OC_Forecast'  ],
                      figsize=(32 , 12)).legend(fontsize=18)
fig = plot.get_figure()
fig.savefig("outputs/bugs_hier_grouping_issue_55_Forecast_world_.png")
