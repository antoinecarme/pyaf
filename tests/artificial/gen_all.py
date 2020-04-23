import os

def createDirIfNeeded(dirname):
    try:
        os.mkdir(dirname);
    except:
        pass

createDirIfNeeded("tests/artificial");


for N in [32, 128, 1024]:
    for transf in ["", "exp" , "log" , "sqrt" , "sqr", "pow3" , "inv" , "diff" , "cumsum"]:
        for trend in ["constant" , "linear" , "poly"]:
            for cycle_length in [0, 5, 7 , 12 , 30]:
                for ar in [0 , 12]:
                    ar_label = "";
                    if(ar > 0):
                        ar_label = ar;
                    lPath = "tests/artificial/transf_" + str(transf);
                    createDirIfNeeded(lPath);
                    lPath = lPath + "/trend_" + str(trend);
                    createDirIfNeeded(lPath);
                    lPath = lPath + "/cycle_" + str(cycle_length);
                    createDirIfNeeded(lPath);
                    lPath = lPath + "/ar_" + str(ar_label);
                    createDirIfNeeded(lPath);
                    for exogc in [0, 20, 100]:
                        name = "test_artificial_" + str(N) + "_" + str(transf) + "_" + str(trend);
                        name = name + "_" + str(cycle_length);
                        name = name + "_" + str(ar_label) + "_" + str(exogc);
                        
                        filename = lPath + "/" + name + ".py";
                        
                        lVariable = "dataset = tsds.generate_random_TS(N = " + str(N);
                        lVariable = lVariable + " , FREQ = 'D', seed = 0, trendtype = \"" ;
                        lVariable = lVariable + str(trend) + "\", cycle_length = " + str(cycle_length);
                        lVariable = lVariable + ", transform = \"" + str(transf);
                        lVariable = lVariable + "\", sigma = 0.0, exog_count = " + str(exogc);
                        lVariable = lVariable + ", ar_order = " + str(ar);
                        lVariable = lVariable + ");";
                        
                        file = open(filename, "w");
                        print("WRTITING_FILE" , filename);
                        file.write("import pyaf.Bench.TS_datasets as tsds\n");
                        file.write("import tests.artificial.process_artificial_dataset as art\n\n");
                        file.write("\n\n\n" + lVariable);
                        file.write("\n\nart.process_dataset(dataset);");
                        file.close();
