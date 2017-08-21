import numpy as np
import pandas as pd


def run_bench_process(arg):
    (lBakcend,  df , col, last_date, H) = arg
    fcst_dict = lBakcend.real_forecast_one_signal(df, col , last_date, H)
    return (arg, fcst_dict)

class cAbstractBackend:
    def __init__(self):
        pass
    
    def forecast_all_signals(self, project_data, last_date, H):
        df = project_data.mVisitsDF
        forecasts = {}
        for col in df.columns:
            if(col != 'Date'):
                fcst_dict = self.real_forecast_one_signal(df, col , last_date, H)
                forecasts[col] = fcst_dict
                print("FORECAST_SIGNAL" , self.__class__.__name__ , col)
        return forecasts

    def forecast_all_signals_multiprocess(self, df , last_date, H):
        import multiprocessing as mp
        nbprocesses = 18
        pool = mp.Pool(processes=nbprocesses, maxtasksperchild=None)
        args = []
        for col in df.columns:
            if(col != 'Date'):
                df1 = df[['Date' , col]]
                args = args + [(self , df1, col, last_date, H)];

        lResults = {};
        i = 1;
        forecasts = {}
        for res in pool.imap(run_bench_process, args):
            signal = res[0][2]
            print("FINISHED_BENCH_FOR_SIGNAL", signal , i , "/" , len(args));
            forecasts[signal] = res[1]
            i = i + 1
            
        pool.close()
        pool.join()

        return forecasts

    def forecast_zero_for_column(self, df, signal, last_date, H):
        fcst_dict = {}
        for h in range(H):
            new_date = np.datetime64(last_date) + np.timedelta64(h + 1, 'D')
            fcst_dict[new_date] = 0   
        return fcst_dict
        

class cZero_Backend (cAbstractBackend):
    def __init__(self):
        cAbstractBackend.__init__(self);
        pass
    
    def real_forecast_one_signal(self, df, signal, last_date, H):
        fcst_dict = self.forecast_zero_for_column(df, signal, last_date, H)
        return fcst_dict

        
class cPyAF_Backend (cAbstractBackend):
    def __init__(self):
        cAbstractBackend.__init__(self);
        pass

    
    def forecast_all_signals(self, project_data , last_date, H):
        return self.forecast_all_signals_multiprocess(project_data.mVisitsDF, last_date, H)
    
    def real_forecast_one_signal(self, df, signal, last_date, H):
        import pyaf.ForecastEngine as autof
        lEngine = autof.cForecastEngine()
        lEngine.mOptions.mAddPredictionIntervals = False
        lEngine.mOptions.mParallelMode = False
        lEngine.mOptions.set_active_transformations(['None', 'Difference' , 'Anscombe'])
        lEngine.mOptions.mMaxAROrder = 16
        
        # lEngine
        df1 = df[['Date' , signal]].fillna(0.0)
        lEngine.train(df1, 'Date' , signal, 1);
        lEngine.getModelInfo();
        # lEngine.standrdPlots()

        df_forecast = lEngine.forecast(iInputDS = df1, iHorizon = H)
        dates = df_forecast['Date'].tail(H).values
        predictions = df_forecast[str(signal) + '_Forecast'].tail(H).values
        # print(dates)
        # print(predictions)
        fcst_dict = {}
        for i in range(H):
            ts = pd.to_datetime(str(dates[i])) 
            date_str = ts.strftime('%Y-%m-%d')
            fcst_dict[date_str] = int(predictions[i])
        print("SIGNAL_FORECAST" , signal, fcst_dict)
        return fcst_dict
        
    

class cPyAF_Backend_2 (cPyAF_Backend):
    def __init__(self):
        cPyAF_Backend.__init__(self);
        pass

    
    def forecast_all_signals(self, project_data , last_date, H):
        df = project_data.mVisitsDF
        df_clean = self.clean_all_signals(df)
        forecasts = self.forecast_all_signals_multiprocess(df_clean, last_date, H)
        for col in df.columns:
            if(col not in df_clean.columns):
                fcst_dict = self.forecast_zero_for_column(df, col, last_date, H)
                forecasts[col] = fcst_dict
        return forecasts

    def is_significant_signal(self, sig):
        lMinVisits = 10
        lMinNonZero = 5
        last_100_values = sig[-100:]
        lNbNonZero = last_100_values[last_100_values > 0].count()
        print("SIGNAL_FILTER_INFO" , sig.name , sig.min() , sig.max() , sig.mean(), sig.std(), lNbNonZero)
        if(sig.max() < lMinVisits):
            return False;
        if(lNbNonZero < lMinNonZero):
            return False
        return True
    
    
    def clean_all_signals(self , df):
        df_out = pd.DataFrame()
        df.info()
        df_out['Date'] = df['Date']
        for col in df.columns:
            if(col != 'Date'):
                if(self.is_significant_signal(df[col])):
                    df_out[col] = df[col]
        df_out.info()
        return df_out
    


class cPyAF_Hierarchical_Backend (cAbstractBackend):
    def __init__(self):
        cAbstractBackend.__init__(self);
        pass

    
    def forecast_all_signals(self, project_data , last_date, H):
        return self.forecast_all_signals_hierarchical(project_data, last_date, H)

    def define_hierarchy(self, project_data):
        rows_list = [];
        for(article_id, article_info) in project_data.mArticleInfo.items():
            (article_id_2, full_name, name, project, access, agent) = (article_info)
            assert(article_id_2 == article_id)
            hier_row = [article_id, access+ '+' + agent , agent, project_data.mName]
            rows_list.append(hier_row)

        lLevels = ['article', 'agent_access' , 'agent' , 'project'];
        lHierarchy = {};
        lHierarchy['Levels'] = lLevels;
        lHierarchy['Data'] = pd.DataFrame(rows_list, columns =  lLevels);
        lHierarchy['Type'] = "Hierarchical";
    
        print(lHierarchy['Data'].head(lHierarchy['Data'].shape[0]));

        return lHierarchy;
    
    
    def forecast_all_signals_hierarchical(self, project_data, last_date, H):
        import pyaf.HierarchicalForecastEngine as hautof

        lEngine = hautof.cHierarchicalForecastEngine()
        lEngine.mOptions.mHierarchicalCombinationMethod = "TD";
        # lEngine
        lEngine.mOptions.mAddPredictionIntervals = False
        # lEngine.mOptions.mParallelMode = False
        lEngine.mOptions.set_active_transformations(['None', 'Difference' , 'Anscombe'])
        lEngine.mOptions.mMaxAROrder = 16

        lHierarchy = self.define_hierarchy(project_data)
        
        # lEngine
        df1 = project_data.mVisitsDF.fillna(0.0)
        lEngine.train(df1, 'Date' , None, 1 , lHierarchy, None);
        lEngine.getModelInfo();
        # lEngine.standrdPlots()

        df_forecast = lEngine.forecast(iInputDS = df1, iHorizon = H)
        print(df_forecast.columns)
        df_forecast.to_csv("hierarchical_td_Forecast.csv")
        dates = df_forecast['Date'].tail(H).values
        forecasts = {}
        for col in df1.columns:
            if(col != 'Date'):
                print("FORECAST_SIGNAL" , self.__class__.__name__ , col)
                predictions = df_forecast[str(col) + '_AHP_TD_Forecast'].tail(H).values
                # print(dates)
                # print(predictions)
                fcst_dict = {}
                for i in range(H):
                    ts = pd.to_datetime(str(dates[i])) 
                    date_str = ts.strftime('%Y-%m-%d')
                    fcst_dict[date_str] = int(predictions[i])
                print("SIGNAL_FORECAST" , col, fcst_dict)
                forecasts[col] = fcst_dict
        
        return forecasts
