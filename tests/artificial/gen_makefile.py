import os
import glob

def mkdir_p(path):
    try:
        os.makedirs(path)
    except:
        pass
    
subdirs = glob.glob("tests/artificial/transf_*");

print("PYTHON=python3\n\n");

lAllTarget = "";
for subdir1 in sorted(subdirs):
    lBase = os.path.basename(subdir1);
    test_target = "";
    for filename in sorted(glob.glob(subdir1 + "/*/*/*/*.py")):
        bn = os.path.basename(filename);
        dirnames = os.path.dirname(filename).split("/");
        logdir = "tests/references/artificial/" + dirnames[2] + "/" + dirnames[3] +  "/" + dirnames[4]
        mkdir_p(logdir)
        logname = bn.replace("/" , "_");
        logname = logname.replace(".py" , ".log");
        
        logfile = "logs/" + logname;
        reflogfile = logdir + "/" + logname;
        difffile = logfile + ".diff"
        print("#PROCESSING FILE : " , filename, bn , logfile);
        print(bn , " : " , "\n\t", "-$(PYTHON) " , filename , " > " , logfile , " 2>&1");
        print("\t", "$(PYTHON) scripts/num_diff.py " , reflogfile , logfile, " > " , difffile);
        print("\t", "tail -10 " ,  difffile, "\n");
        test_target = bn + " " + test_target;

    lAllTarget = lAllTarget + " " + lBase;
    print("\n\n", lBase , ": ", test_target, "\n" , "\n");

print("\n# ********************************************** \n");

print("all: " , lAllTarget , "\n\t\n");
