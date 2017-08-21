import pyaf.Bench.web_traffic.Forecaster_Backends as be
import pandas as pd
import numpy as np
import pickle
import datetime
                
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
        pkl_file = open(self.mDataDirectory + "/" + project + ".pkl", 'rb')
        lProjectData = pickle.load(pkl_file)
        pkl_file.close()
        return lProjectData                
        
    def read_keys_file_if_needed(self):
        if(self.mShortednedIds is None):
            self.mKeysDF = pd.read_csv(self.mDataDirectory + "/" + self.mKeysFileName)
            self.mKeysDF[['Id']].to_csv('tmp/ids.txt' , index=False)
            self.mShortednedIds = {}
            for row in self.mKeysDF.itertuples():
                self.mShortednedIds[row[1]] = row[2]
            pass

    def get_shortened_id(self, article_full_name, date):
        lKey = article_full_name + "_" + str(date)
        lShort = self.mShortednedIds.get(lKey)
        if(lShort is None):
            print("SHORT_ID_NOT_FOUND" , article_full_name , date, lKey)
        assert(lShort is not None)
        return lShort

    def forecast(self, projects, iLastDate, iHorizon):
        self.read_keys_file_if_needed()
        lFactory = cForecaster_Backend_Factory()
        self.mBackend = lFactory.create(self.mBackendName)

        outputs = []
        for project in projects:
            print("FORECASTING_WIKIPEDIA_PROJECT" , project)
            project_data = self.load_data(project)
            forecasts = self.mBackend.forecast_all_signals(project_data, iLastDate , iHorizon)
            for col in sorted(forecasts.keys()):
                lInfo = project_data.mArticleInfo[col]
                lFullName = lInfo[1]
                fcst_dict = forecasts[col]
                for k in sorted(fcst_dict.keys()):
                    v = fcst_dict[k]
                    lShortened = self.get_shortened_id(lFullName , k)
                    outputs.append([lShortened , v])
        self.save_submission(outputs)

    def save_submission(self, outputs):        
        lFilename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        lFilename = "tmp/submit_" + str(self.mBackendName) + "_" + lFilename + ".csv.gz"
        output_df = pd.DataFrame(outputs , columns = ["Id" , "Visits"]);
        print("WRITING_OUTPUT" , lFilename)
        output_df.to_csv(lFilename, compression='gzip', index=False)
        print("WRITING_OUTPUT_OK" , lFilename)
        self.check_submission(lFilename)
        
    def check_submission(self , filename):
        print("SUBMIT_CHECK_START")
        lSubmitDF = pd.read_csv(filename)
        visited_ids = {}
        for row in lSubmitDF.itertuples():
            visited_ids[row[1]] = True
        # check that all short ids are present in the submission
        for (page, short_id) in self.mShortednedIds.items():
            if(visited_ids.get(short_id) is None):
                print("SUBMIT_CHECK_FAILED")
                print("MISSING_PREDICTION_FOR" , short_id, page)
                # assert(0)
    
        print("SUBMIT_CHECK_OK")
