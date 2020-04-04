import os

# exit()

def createDirIfNeeded(dirname):
    try:
        os.mkdir(dirname);
    except:
        pass

createDirIfNeeded("tests/artificial");

def gen_file(cyc , freq, nbrows):
    lDir = "tests/periodicities/" + freq[1]; 
    createDirIfNeeded(lDir)
    filename = lDir + "/Cycle_" + str(freq[1]) + "_" +str(nbrows) + "_" + str(freq[0]) + "_" + str(cyc) + ".py";
    file = open(filename, "w");
    print("WRTITING_FILE" , filename);
    file.write("import tests.periodicities.period_test as per\n\n");
    file.write("per.buildModel((" + str(cyc) + " , '" + str(freq[0]) + "' , " + str(nbrows) + "));\n\n");
    file.close();
    


def generate_all():

    lCycles = [ 5 , 7 , 12 , 15, 24, 30 , 60, 120, 360];
    lFreqs = [('S' , 'Second'),
              ('T', 'Minute'),
              ('H' , 'Hour'),
              ('BH', 'Business_Hour'),
              ('D' , 'Day'),
              ('B', 'Business_Day'),
              ('W' , 'Week'),
              ('SM', 'Semi_Month'),
              ('M' , 'Month')]

    for cyc in lCycles:
        for freq in lFreqs:
            lRows = [25, 50, 100, 200, 400, 1600, 3200];
            if(freq[0] == 'W'):
                lRows = [25, 50, 100, 200, 400, 1600];
            if(freq[0] == 'SM'):
                lRows = [25, 50, 100, 200, 400];
            if(freq[0] == 'M'):
                lRows = [25, 50, 100, 200, 400];
            for nbrows in lRows:
                gen_file(cyc , freq, nbrows);

    


generate_all();
