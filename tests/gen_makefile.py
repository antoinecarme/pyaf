import os
import glob

str1 = "artificial bugs exog hierarchical model_control perf svr transformations bench func neuralnet real-life  time_res";
subdirs = str1.split();

print("PYTHON=python3\n\n");

for subdir1 in subdirs:
    test_target = "";
    for filename in glob.glob("tests/" + subdir1 + "/*.py"):
        bn = subdir1 + "/" + os.path.basename(filename);
        logfile = bn.replace("/" , "_");
        logname = logfile.replace(".py" , ".log");
        logfile = "logs/" + logname;
        reflogfile = "tests/references/" + logname;
        difffile = logfile + ".diff"
        # print("#PROCESSING FILE : " , filename, bn , logfile);
        
        print(bn , " : " , "\n\t", "-$(PYTHON) " , filename , " > " , logfile , " 2>&1");
        print("\t", "-diff " , logfile , reflogfile , " > " , difffile, "\n");
        print("\t", "cat " ,  difffile, "\n");
        
        
        test_target = bn + " " + test_target;

    print("\n\n", subdir1, ": ", test_target, "\n" , "\n");

print("\n# ********************************************** \n");

print("all: " , str1 , "\n\t\n");
