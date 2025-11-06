
FILES = ['cran_tsoutliers_hicp_000000.csv', 'cran_tsoutliers_ipi_Greece.csv', 'cran_tsoutliers_ipi_Luxembourg.csv', 'cran_tsoutliers_hicp_IGXE00.csv', 'cran_tsoutliers_hicp_011200.csv', 'cran_tsoutliers_hicp_011300.csv', 'cran_tsoutliers_hicp_010000.csv', 'cran_tsoutliers_ipi_Belgium.csv', 'cran_tsoutliers_ipi_Slovakia.csv', 'cran_tsoutliers_bde9915_euprin.csv', 'cran_fpp2_gold.csv', 'cran_tsoutliers_ipi_Netherlands.csv', 'cran_tsoutliers_ipi_Malta.csv', 'cran_tsoutliers_hicp_SERV00.csv', 'cran_tsoutliers_ipi_Latvia.csv', 'cran_tsoutliers_ipi_Finland.csv', 'cran_tsoutliers_bde9915_metipi.csv', 'cran_tsoutliers_hicp_011000.csv', 'cran_tsoutliers_ipi_Slovenia.csv', 'cran_tsoutliers_ipi_Estonia.csv', 'cran_tsoutliers_hicp_011600.csv', 'cran_tsoutliers_hicp_011700.csv', 'cran_tsoutliers_ipi_Portugal.csv', 'cran_tsoutliers_ipi_Spain.csv', 'cran_tsoutliers_ipi_Italy.csv', 'cran_tsoutliers_ipi_France.csv', 'cran_tsoutliers_hicp_NRGY00.csv', 'cran_tsoutliers_bde9915_gipi.csv', 'cran_tsoutliers_ipi_Austria.csv', 'cran_tsoutliers_hicp_FOODPR.csv', 'cran_tsoutliers_hicp_FOODUN.csv', 'cran_tsoutliers_ipi_Germany.csv', 'cran_tsoutliers_ipi_Cyprus.csv']

URI = "https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/outliers/"

for f in FILES:
    print("GENERATING_TEST", f)
    pf = f.replace(".csv", ".py")
    pf = "test_" + pf
    with open("tests/outliers/" + pf, "w") as f1:
        f1.write("import tests.outliers.tsoutliers_generic as outlier_test\n\n\n")
        if("fpp2_gold" in f):
            f1.write('outlier_test.process_tsoutliers_signal("' + URI + f + '", iconv_time = False)\n')
        else:
            f1.write('outlier_test.process_tsoutliers_signal("' + URI + f + '", iconv_time = True)\n')
            
