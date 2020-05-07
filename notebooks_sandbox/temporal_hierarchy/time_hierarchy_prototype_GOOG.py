# %matplotlib inline
import pyaf
import datetime

goog_link = 'https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/YahooFinance/nasdaq/yahoo_GOOG.csv'
    
import pandas as pd
goog_dataframe = pd.read_csv(goog_link);
goog_dataframe['Date'] = goog_dataframe['Date'].apply(lambda x : datetime.datetime.strptime(x, "%Y-%m-%d"))
goog_dataframe.sort_values(by = 'Date' , ascending=True, inplace=True)
goog_dataframe.tail()

lHierarchy = {};
lHierarchy['Levels'] = None;
lHierarchy['Data'] = None;
lHierarchy['Groups']= {};

lHierarchy['Periods']= ["D" , "W" , "Q"]

lHierarchy['Type'] = "Temporal";


# create a model to plot the hierarchy.
import pyaf.HierarchicalForecastEngine as hautof
lEngine = hautof.cHierarchicalForecastEngine()


lSignalHierarchy = lEngine.plot_Hierarchy(goog_dataframe , "Date", "Close", 100, 
                                          lHierarchy, None);

print(lSignalHierarchy.__dict__)


# create a hierarchical model and train it
import pyaf.HierarchicalForecastEngine as hautof

lEngine = hautof.cHierarchicalForecastEngine()
lEngine.mOptions.mNbCores = 1
lEngine.mOptions.mHierarchicalCombinationMethod = ["BU" , 'TD' , 'MO' , 'OC'];

lDateColumn = "Date"
lSignalVar = "Close";

#
N = goog_dataframe.shape[0];
H = 4;
train_df = goog_dataframe


lSignalHierarchy = lEngine.train(train_df , lDateColumn, lSignalVar, 100, lHierarchy, None);
lEngine.getModelInfo();
