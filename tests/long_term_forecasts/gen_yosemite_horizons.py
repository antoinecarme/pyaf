lHorizons = [1] + [h for h in range(100, 1000+1, 100)]

for H in lHorizons:
    lDir = "tests/long_term_forecasts" 
    filename = lDir + "/test_yosemite_temps_Horizon_" + str(H) + ".py";
    file = open(filename, "w");
    print("WRTITING_FILE" , filename);
    file.write("import tests.long_term_forecasts.test_yosemite_temps as yos\n\n");
    file.write("yos.buildModel(" + str(H) + ");\n\n");
    file.close();
