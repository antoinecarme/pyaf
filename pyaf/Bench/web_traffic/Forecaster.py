import pyaf.Bench.web_traffic.Forecaster_Backends as be
import pandas as pd
import numpy as np
import pickle
import datetime
import time

import pyaf.Bench.web_traffic.Bench as wtbench
logger = wtbench.get_bench_logger()

class cForecaster_Backend_Factory:
    def __init__(self):
        pass 
    
    def create(self, backend_name = None):
        if(backend_name == "pyaf_default"):
            return be.cPyAF_Backend()
        if(backend_name == "pyaf_default_clean"):
            return be.cPyAF_Backend_2()
        if(backend_name == "pyaf_hierarchical_top_down"):
            return be.cPyAF_Hierarchical_Backend()
        return be.cZero_Backend()

class cProjectForecaster:
    def __init__(self):
        self.mShortednedIds = None
        self.mBackendName = None
        self.mKeysFileName = None
        self.mDataDirectory = None
        
    def load_data(self, project):
        logger.info("LOADING_PROJECT_TRAINING_DATA_START " + project)
        pkl_file = open(self.mDataDirectory + "/" + project + ".pkl", 'rb')
        lProjectData = pickle.load(pkl_file)
        pkl_file.close()
        lProjectData.dump()
        logger.info("LOADING_PROJECT_TRAINING_DATA_END " + project)
        return lProjectData                
        
    def read_keys_file_if_needed(self):
        if(self.mShortednedIds is None):
            logger.info("LOADING_SHORTENED_IDS_START")
            self.mKeysDF = pd.read_csv(self.mDataDirectory + "/" + self.mKeysFileName)
            # self.mKeysDF[['Id']].to_csv('tmp/ids.txt' , index=False)
            self.mShortednedIds = {}
            for row in self.mKeysDF.itertuples():
                self.mShortednedIds[row[1]] = row[2]
            logger.info("LOADING_SHORTENED_IDS_END " + str(len(self.mShortednedIds)))
            pass

    def get_shortened_id(self, article_full_name, date):
        lKey = article_full_name + "_" + str(date)
        lShort = self.mShortednedIds.get(lKey)
        if(lShort is None):
            logger.info("SHORT_ID_NOT_FOUND " + str(article_full_name)  + " " + str(date) + " " +  lKey)
        assert(lShort is not None)
        return lShort

    def forecast(self, projects, iLastDate, iHorizon):
        self.read_keys_file_if_needed()
        lFactory = cForecaster_Backend_Factory()
        self.mBackend = lFactory.create(self.mBackendName)

        outputs = []
        for project in projects:
            start_time = time.time()
            logger.info("FORECASTING_WIKIPEDIA_PROJECT_START " + project)
            project_data = self.load_data(project)
            forecasts = self.mBackend.forecast_all_signals(project_data, iLastDate , iHorizon)
            for col in sorted(forecasts.keys()):
                lInfo = project_data.mArticleInfo[col]
                lFullName = lInfo[1]
                fcst_dict = forecasts[col]
                for k in sorted(fcst_dict.keys()):
                    v = fcst_dict[k]
                    v = int(v) if (v>0) else 0
                    lShortened = self.get_shortened_id(lFullName , k)
                    outputs.append([lShortened , v])
                    logger.info("FORECASTING_WIKIPEDIA_PROJECT_DETAIL " + " " + str(col) + " " + str(k) + " " +\
                                lFullName + " " + str(k) + " " + lShortened + " "  + str(v))
            self.save_submission(outputs , project)
            logger.info("FORECASTING_WIKIPEDIA_PROJECT_END_TIME_IN_SECONDS " + project + " " + str(time.time() - start_time))
            

    def save_submission(self, outputs, project):        
        lFilename = datetime.datetime.now().strftime("%Y_%m_%d")
        lFilename = "tmp/submit_" + str(project) + "_" + str(self.mBackendName) + "_" + lFilename + ".csv.gz"
        output_df = pd.DataFrame(outputs , columns = ["Id" , "Visits"]);
        logger.info("WRITING_OUTPUT " + lFilename)
        output_df.to_csv(lFilename, compression='gzip', index=False)
        logger.info("WRITING_OUTPUT_OK " + lFilename)
        self.check_submission(lFilename , project)
        
    def check_submission(self , filename, project):
        logger.info("SUBMIT_CHECK_START " + project)
        lSubmitDF = pd.read_csv(filename)
        visited_ids = {}
        for row in lSubmitDF.itertuples():
            visited_ids[row[1]] = True
        # print(list(visited_ids.keys())[:10])
        # print(list(self.mShortednedIds.items())[:10])
        # check that all short ids are present in the submission
        for (page, short_id) in self.mShortednedIds.items():
            if(project in page and visited_ids.get(short_id) is None):
                logger.info("SUBMIT_CHECK_FAILED " + project)
                logger.info("MISSING_PREDICTION_FOR " + short_id + " " + page)
                # assert(0)
    
        logger.info("SUBMIT_CHECK_OK " + project)
