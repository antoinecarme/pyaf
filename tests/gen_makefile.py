import os
import glob


def add_makefile_entry(subdir1):
    test_target = "";
    for filename in sorted(glob.glob("tests/" + subdir1 + "/*.py")):
        lShortName = os.path.basename(filename);
        if(not lShortName.lower().startswith("gen_all") and
           not lShortName.lower().startswith("gen_makefile") and
           not "prototyp" in lShortName.lower() and
           not "_slow_mode" in lShortName.lower()):
            bn = subdir1 + "/" + lShortName;
            logfile = bn.replace("/" , "_");
            logname = logfile.replace(".py" , ".log");
            logfile = "logs/" + logname;
            reflogfile = "tests/references/" + logname;
            difffile = logfile + ".diff"
            # print("#PROCESSING FILE : " , filename, bn , logfile);
        
            print(bn , " : " , "\n\t", "$(PYTHON) " , filename , " > " , logfile, " 2>&1");
            print("\t", "$(PYTHON) scripts/num_diff.py " , reflogfile , logfile, " > " , difffile);
            print("\t", "tail -10 " ,  difffile, "\n");
                
            test_target = bn + " " + test_target;

    return test_target;


str1 = "artificial basic_checks bugs cross_validation croston exog expsmooth func HeartRateTimeSeries heroku hierarchical  HourOfWeek model_control perf svr transformations  neuralnet real-life  time_res perfs demos xgb xeon-phi-parallel sampling temporal_hierarchy WeekOfMonth missing_data";
str1 = str1 + " probabilistic_forecasting"
subdirs = str1.split();

print("PYTHON=python3\n\n");

for subdir1 in sorted(subdirs):
    test_target = add_makefile_entry(subdir1)
    if(subdir1 == "bugs"):
        bugdirs = glob.glob("tests/bugs/*")
        bugdirs1 = [dir1.replace("tests/" , "") for dir1 in bugdirs]
        for dir1 in bugdirs1:
            test_target = test_target + add_makefile_entry(dir1)
    print("\n\n", subdir1, ": ", test_target, "\n" , "\n");

print("\n# ********************************************** \n");

print("all: " , str1 , "\n\t\n");

str2 = "demos basic_checks cross_validation croston exog heroku hierarchical model_control perfs svr transformations func real-life  time_res xgb sampling HourOfWeek WeekOfMonth missing_data";

print("build-test : " , str2 , "\n\t\n");
