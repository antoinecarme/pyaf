import os
import glob

test_target = "";
for filename in glob.glob('tests/*.py'):
    bn = os.path.basename(filename);
    logfile = "logs/" + bn.replace(".py" , ".log");
    print("#PROCESSING FILE : " , filename, bn);
    
    print(bn , " : " , "\n\t", "-ipython3" , filename , " > " , logfile , " 2>&1");
    test_target = bn + " " + test_target;

print("\n\ntests: ", test_target, "\n" , "\n");

print("\n# ********************************************** \n");

