import numpy as np

def gen_all_ar_speed_tests():
    lDir = "tests/perf"
    for order in np.arange(0, 1000, 50):
        filename = lDir + "/test_ozone_ar_speed_order_" + str(order) + ".py";
        with open(filename, "w") as outfile:
            print("WRTITING_FILE" , filename)
            outfile.write("import tests.perf.test_ozone_ar_speed_many as gen\n\n")
            outfile.write("gen.run_test(" + str(order) + ")\n\n")

# gen_all_ar_speed_tests()
