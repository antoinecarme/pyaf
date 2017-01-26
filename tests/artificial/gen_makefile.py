import os
import glob

subdirs = glob.glob("tests/artificial/transf_*");

print("PYTHON=python3\n\n");

lAllTarget = "";
for subdir1 in sorted(subdirs):
    lBase = os.path.basename(subdir1);
    test_target = "";
    for filename in sorted(glob.glob(subdir1 + "/*/*/*/*.py")):
        bn = os.path.basename(filename);
        logfile = bn.replace("/" , "_");
        logfile = "logs/" + logfile.replace(".py" , ".log");
        print("#PROCESSING FILE : " , filename, bn , logfile);
        
        print(bn , " : " , "\n\t", "-$(PYTHON) " , filename , " > " , logfile , " 2>&1");
        test_target = bn + " " + test_target;

    lAllTarget = lAllTarget + " " + lBase;
    print("\n\n", lBase , ": ", test_target, "\n" , "\n");

print("\n# ********************************************** \n");

print("all: " , lAllTarget , "\n\t\n");
