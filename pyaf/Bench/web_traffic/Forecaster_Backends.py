import numpy as np
import pandas as pd
import pyaf.Bench.web_traffic.Bench as be

logger = be.get_bench_logger()


def run_bench_process(arg):
    (lBakcend,  df , cols, last_date, H) = arg
    res = {}
    for col in cols:
        fcst_dict = lBakcend.real_forecast_one_signal(df, col , last_date, H)
        res[col] = fcst_dict
    return (arg, res)

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
                # logger.info("FORECAST_SIGNAL " + str([self.__class__.__name__ , col]))
        return forecasts

    def forecast_all_signals_multiprocess(self, df , last_date, H):
        import multiprocessing as mp
        nbprocesses = 18
        pool = mp.Pool(processes=nbprocesses, maxtasksperchild=None)
        args = []
        cols = []
        # print(df.columns)
        for col in df.columns:
            if(col != 'Date'):
                cols = cols + [col]
                if(len(cols) > 50):
                    # print(cols , ['Date'] + cols)
                    df1 = df[['Date'] + cols]
                    args = args + [(self , df1, cols, last_date, H)];
                    cols = []
        if(len(cols) > 0):
            # print(cols , ['Date'] + cols)
            df1 = df[['Date'] + cols]
            args = args + [(self , df1, cols, last_date, H)];
            cols = []
            

        i = 1;
        forecasts = {}
        for res in pool.imap(run_bench_process, args):
            signals = res[0][2]
            for sig in signals:
                logger.info("FINISHED_BENCH_FOR_SIGNAL " + str(sig)  + " " +  str(i) + "/" + str(len(df.columns)));
                forecasts[sig] = res[1][sig]
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
        # logger.info(dates)
        # logger.info(predictions)
        fcst_dict = {}
        for i in range(H):
            ts = pd.to_datetime(str(dates[i])) 
            date_str = ts.strftime('%Y-%m-%d')
            fcst_dict[date_str] = int(predictions[i])
        logger.info("SIGNAL_FORECAST " +  str(signal) + " " +  str(fcst_dict))
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
        logger.info("SIGNAL_FILTER_INFO " + str([sig.name , sig.min() , sig.max() , sig.mean(), sig.std(), lNbNonZero]))
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
        self.mOtherArticleId = -1
        self.mMaxNbKept = 100
        pass

    
    def forecast_all_signals(self, project_data , last_date, H):
        return self.forecast_all_signals_hierarchical(project_data, last_date, H)

    def reduce_articles(self, project_data, articles):
        sums = {}
        total_sum = 0;
        for article_id in articles:
            sums[article_id] = project_data.mVisitsDF[article_id].fillna(0.0).sum()
            total_sum = total_sum + sums[article_id]
        logger.info("SUMS_BEFORE_SORTING " + str((list(sums)[0:self.mMaxNbKept] , total_sum)))
        # order by total count
        sorted_sums = sorted(sums.items(), key=lambda x: x[1], reverse=True)
        logger.info("SUMS_AFTER_SORTING " + str(sorted_sums[:self.mMaxNbKept]))
        result = []
        cum_sum = 0;
        threshold = 0.95 * total_sum
        for (article_id , count) in sorted_sums[:self.mMaxNbKept]:
            if(cum_sum <= threshold):
                result.append(article_id)
            cum_sum = cum_sum + sums[article_id]
        logger.info("SUMS_AFTER_SORTING_2 " + str((threshold, cum_sum , cum_sum / total_sum, len(articles), len(result))))
        return result
        

    def analyze_signals(self, project_data):
        self.base_level = {}
        for(article_id, article_info) in project_data.mArticleInfo.items():
            (article_id_2, full_name, name, project, access, agent) = (article_info)
            acc_ag_key = access+ '+' + agent
            if(self.base_level.get(acc_ag_key) is None):
                self.base_level[acc_ag_key] = set()
            self.base_level[acc_ag_key].add(article_id)
        self.clean_base_level = {}
        self.nb_discarded = {}
        for (acc_ag_key, articles) in self.base_level.items():
            important_articles = self.reduce_articles(project_data , articles)
            self.clean_base_level[acc_ag_key] = important_articles
            self.nb_discarded[acc_ag_key] = len(articles) - len(important_articles)
        pass
    
    def define_hierarchy(self, project_data):
        self.analyze_signals(project_data)
        rows_list = [];
        self.mHierarchyVisits = pd.DataFrame()
        self.mHierarchyClass = {}
        self.mHierarchyVisits['Date'] =  project_data.mVisitsDF['Date']
        for(article_id, article_info) in project_data.mArticleInfo.items():
            (article_id_2, full_name, name, project, access, agent) = (article_info)
            assert(article_id_2 == article_id)
            acc_ag_key = access+ '+' + agent
            series = project_data.mVisitsDF[article_id]
            if(article_id_2 not in self.clean_base_level[acc_ag_key]):
                article_id_2 = self.mOtherArticleId
                self.mHierarchyClass[article_id] = acc_ag_key
                if(article_id_2 not in self.mHierarchyVisits.columns):
                    self.mHierarchyVisits[article_id_2] = series
                    hier_row = [article_id_2, access+ '+' + agent , agent, project_data.mName]
                else:
                    self.mHierarchyVisits[article_id_2] = self.mHierarchyVisits[article_id_2] + series
            else:
                self.mHierarchyVisits[article_id_2] = series
                hier_row = [article_id_2, access+ '+' + agent , agent, project_data.mName]
                
            rows_list.append(hier_row)

        lLevels = ['article', 'agent_access' , 'agent' , 'project'];
        lHierarchy = {};
        lHierarchy['Levels'] = lLevels;
        lHierarchy['Data'] = pd.DataFrame(rows_list, columns =  lLevels);
        lHierarchy['Type'] = "Hierarchical";
    
        logger.info(str(lHierarchy['Data'].head(lHierarchy['Data'].shape[0])));
        logger.info(str(lHierarchy['Data'].info()));
        logger.info(str(self.mHierarchyVisits.info()));

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
        df1 = self.mHierarchyVisits.fillna(0.0)
        lEngine.train(df1, 'Date' , None, 1 , lHierarchy, None);
        lEngine.getModelInfo();
        # lEngine.standrdPlots()

        df_forecast = lEngine.forecast(iInputDS = df1, iHorizon = H)
        logger.info(str(df_forecast.columns))
        df_forecast.to_csv("hierarchical_td_Forecast.csv")
        dates = df_forecast['Date'].tail(H).values
        forecasts = {}
        for col in project_data.mVisitsDF.columns:
            if(col != 'Date'):
                logger.info("FORECAST_SIGNAL "  +  str([self.__class__.__name__ , col]))
                predictions = None
                if(col in self.mHierarchyVisits.columns):
                    predictions = df_forecast[str(col) + '_AHP_TD_Forecast'].tail(H).values
                else:
                    nb_other = self.nb_discarded[self.mHierarchyClass[col]]
                    predictions = df_forecast[str(self.mOtherArticleId) + '_AHP_TD_Forecast'] / nb_other
                    predictions = predictions.tail(H).values
                    
                # logger.info(dates)
                # logger.info(predictions)
                fcst_dict = {}
                for i in range(H):
                    ts = pd.to_datetime(str(dates[i])) 
                    date_str = ts.strftime('%Y-%m-%d')
                    fcst_dict[date_str] = int(predictions[i])
                logger.info("SIGNAL_FORECAST "  + str([col, fcst_dict]))
                forecasts[col] = fcst_dict
        
        return forecasts
