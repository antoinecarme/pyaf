




def gen_all_air_passengers():
    lDir = "tests/missing_data"
    for iTimeImp in [None , "DiscardRow", "Interpolate"]:
        for iSigImp in [None , "DiscardRow", "Interpolate", "Mean", "Median" , "Constant" , "PreviousValue"]:
            filename = lDir + "/test_missing_data_air_passengers_" + str(iTimeImp) + "_" + str(iSigImp) + ".py";
            with open(filename, "w") as outfile:
                print("WRTITING_FILE" , filename)
                outfile.write("import tests.missing_data.test_missing_data_air_passengers_generic as gen\n\n")
                outfile.write("gen.test_air_passengers_missing_data" + str((iTimeImp , iSigImp)) + "\n")

# gen_all_air_passengers()
