import os
import glob

str1 = "artificial";
subdirs = str1.split();

for subdir1 in subdirs:
    test_target = "";
    for filename in glob.glob("tests/" + subdir1 + "/*/*/*/*/*.py"):
        bn = os.path.basename(filename);
        logfile = bn.replace("/" , "_");
        logfile = "logs/" + logfile.replace(".py" , ".log");
        print("#PROCESSING FILE : " , filename, bn , logfile);
        
        print(bn , " : " , "\n\t", "-ipython3" , filename , " > " , logfile , " 2>&1");
        test_target = bn + " " + test_target;

    print("\n\n", subdir1, ": ", test_target, "\n" , "\n");

print("\n# ********************************************** \n");

print("all: " , str1 , "\n\t\n");
