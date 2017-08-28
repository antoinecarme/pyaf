import os
import glob

base_targets = []

def add_makefile_entry(subdir1):
    test_target = "";
    for filename in sorted(glob.glob(subdir1 + "/*.py")):
        lShortName = os.path.basename(filename);
        bn = lShortName;
        logfile = bn.replace("/" , "_");
        logname = logfile.replace(".py" , ".log");
        logfile = "logs/" + logname;
        print(bn , " : " , "\n\t", "-$(PYTHON) " , filename , " > " , logfile, " 2>&1");                
        test_target = bn + " " + test_target;
        base_targets.append(bn)
    return test_target;


subdirs = glob.glob("tests/bench/web_traffic_jobs/*.org");

print("PYTHON=python3\n\n");

all_tgt = ""

for subdir1 in sorted(subdirs):
    test_target = add_makefile_entry(subdir1)
    lShortName = os.path.basename(subdir1);
    print("\n\n", lShortName, ": ", test_target, "\n" , "\n");
    all_tgt = all_tgt + " " + lShortName
    
print("\n# ********************************************** \n");

print("all: " , all_tgt , "\n\t\n");


be_targets = {}
for be in ['pyaf_default', "pyaf_hierarchical_top_down"]:
    be_targets[be] = []
    for tgt in base_targets:
        if((be + ".py") in tgt):
            be_targets[be].append(tgt)


for be in be_targets.keys():
    tgt = " ".join(be_targets[be])
    print(be + ":" + tgt + "\n\t\n")
