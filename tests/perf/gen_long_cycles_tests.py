import numpy as np

def gen_all_long_cycles_tests():
    lDir = "tests/perf"
    for nbrows in range(1000,42000, 10000):
        for cyc in [ k for k in range(20 ,500, 60)]:
            filename = lDir + "/test_long_cycles_nbrows_cycle_length_" + str(nbrows) + "_" + str(cyc) + ".py";
            with open(filename, "w") as outfile:
                print("WRTITING_FILE" , filename)
                outfile.write("import tests.perf.test_cycles_full_long_long as gen\n\n")
                outfile.write("gen.test_nbrows_cycle(" + str(nbrows) + " , " + str(cyc) + ")\n\n")

# gen_all_long_cycles_tests()
