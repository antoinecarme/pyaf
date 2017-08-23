import os

def createDirIfNeeded(dirname):
    try:
        os.makedirs(dirname);
    except:
        pass


PROJECTS = ['commons.wikimedia.org', 'de.wikipedia.org',
            'es.wikipedia.org', 'www.mediawiki.org', 
            'fr.wikipedia.org', 'ru.wikipedia.org',
            'en.wikipedia.org', 'zh.wikipedia.org',
            'ja.wikipedia.org']


BACKENDS = ["zero" ,  "pyaf_default",  "pyaf_default_clean",
            "pyaf_hierarchical_top_down"];



def create_job(project , backend):
    directory = "tests/bench/web_traffic_jobs" + "/" + project
    createDirIfNeeded(directory);
    filename= directory + "/test_web_traffic_" + str(project) + "_" + str(backend) + ".py";
    file = open(filename, "w");
    print("WRTITING_FILE" , filename);
    file.write("import pyaf.Bench.web_traffic.Forecaster as fo\n\n\n");
    file.write("PROJECTS = ['" + project + "']\n\n");
    file.write("data_dir = 'data/web-traffic-time-series-forecasting'\n\n");
    file.write("lForecaster = fo.cProjectForecaster()\n");
    file.write("lForecaster.mDataDirectory = data_dir\n");

    file.write("lForecaster.mBackendName = '" + backend + "'\n");
    file.write("lForecaster.mKeysFileName = 'key_1.csv.zip'\n");

    file.write("last_date = '2016-12-31'\n");
    file.write("horizon = 60\n");
    file.write("lForecaster.mKeysFileName = 'key_1.csv.zip'\n");
    
    file.write("lForecaster.forecast(PROJECTS, last_date , horizon)\n");
    file.close();


for proj in PROJECTS:
    for be in BACKENDS:
        create_job(proj , be)
        
