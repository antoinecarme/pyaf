import numpy as np
import pandas as pd
import pickle
import datetime
import sys


def get_bench_logger():
    import logging;
    logger = logging.getLogger('pyaf.bench');
    if(logger.handlers == []):
        import logging.config
        logging.basicConfig(level=logging.INFO)        
    return logger;

logger = get_bench_logger()

class cProjectData:
    def __init__(self):
        self.mName = None
        self.mAgents = set()
        self.mAccess = set()
        self.mArticleNames = set()
        self.mArticleInfo = {}
        self.mVisitsDF = pd.DataFrame()

    
    def set_date(self, date_var):
        self.mVisitsDF['Date'] = date_var
    
    def add_article(self, article_id, full_name, name, project, access, agent, article_series):
        if(np.random.random_sample() > 10.01):
            return
        self.mAccess.add(access)
        self.mAgents.add(agent)
        self.mArticleNames.add(name)
        self.mArticleInfo[article_id] = (article_id, full_name, name, project, access, agent)
        self.mVisitsDF[article_id] = article_series

    def dump(self):
        logger.info("PROJECT_DUMP_START " + str(self.mName))
        logger.info("PROJECT_DUMP_AGENTS " +str(self.mAgents))
        logger.info("PROJECT_DUMP_ACCESS "  + str(self.mAccess))
        logger.info("PROJECT_DUMP_NUMBER_OF_ARTICLES " + str(len(self.mArticleInfo)))
        
        lIds = list(self.mArticleInfo.keys())
        lArticles = lIds[0:5] + lIds[-5:]
        logger.info("PROJECT_DUMP_ARTICLE_NAMES " + str([( k , self.mArticleInfo[k][2]) for k in lArticles]))
        logger.info("PROJECT_DUMP_ARTICLE_PROJECTS" + str([( k , self.mArticleInfo[k][3]) for k in lArticles]))
        df = self.mVisitsDF[['Date'] + lArticles]
        print(df.info())
        print(df.describe())
        print(df.head())
        print(df.tail())
        logger.info("PROJECT_DUMP_END " + self.mName)
        sys.stdout.flush()
        
class cDataExtractor:
    def __init__(self):
        self.mProjectDataByName = None
    
    def read_projects_data(self, filename):
        self.mOriginalData = pd.read_csv(filename)
        self.mOriginalDataTransposed = self.mOriginalData.drop('Page' , axis=1).transpose()
        self.mOriginalDataTransposed = self.mOriginalDataTransposed.reset_index()
        self.mOriginalDataTransposed.sort_values(by='index', inplace=True)
        self.mDate_var = self.mOriginalDataTransposed['index'].apply(lambda x : datetime.datetime.strptime(x, "%Y-%m-%d"))
        self.parse_project_data()
        for (name, project) in self.mProjectDataByName.items():
            project.dump()
        pass
    
    def parse_project_data(self):
        pages_dict = self.mOriginalData['Page'].to_dict()
        self.mProjectDataByName = {}
        for (k,v) in pages_dict.items():
            series = self.mOriginalDataTransposed[k]
            words = v.split('_')
            (name, project, access, agent) = ("_".join(words[:-3]) , words[-3] , words[-2] , words[-1])
            lProjectData = self.mProjectDataByName.get(project)
            if(lProjectData is None):
                lProjectData = cProjectData()
                lProjectData.set_date(self.mDate_var)
                lProjectData.mName = project
                self.mProjectDataByName[project] = lProjectData
            lProjectData.add_article(k, v, name, project, access, agent, series)
            
    
    def save_project_data(self , dest_dir):
        for (k,v) in self.mProjectDataByName.items():
            output = open(dest_dir + "/" + k + '.pkl', 'wb')
            pickle.dump(v, output)
            output.close()
        
