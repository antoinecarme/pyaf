import os

def createDirIfNeeded(dirname):
    try:
        os.mkdir(dirname);
    except:
        pass

createDirIfNeeded("tests/exog/random");

N = 600
for n in [600, 300, 150, 75, 32, 16, 8]:
    for nbex in [10 , 20 , 40 , 80, 160, 320, 640, 1280 , 2000]:
        filename= "tests/exog/random/random_exog_" + str(n) + "_" + str(nbex) + ".py";
        file = open(filename, "w");
        print("WRTITING_FILE" , filename);
        file.write("import tests.exog.test_random_exogenous as testrandexog\n");
        file.write("\n\ntestrandexog.test_random_exogenous( " + str(n) + "," + str(nbex) + ");");
        file.close();
