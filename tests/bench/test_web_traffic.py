
import pyaf.Bench.web_traffic.Bench as bench

data_dir = "data/web-traffic-time-series-forecasting"
lExtractor = bench.cDataExtractor()
lExtractor.read_projects_data(data_dir + "/train_1.csv.zip")
lExtractor.save_project_data(data_dir)
print(list(lExtractor.mProjectDataByName.keys()))
