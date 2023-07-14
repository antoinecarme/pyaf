import tests.outliers.tsoutliers_generic as outlier_test


outlier_test.process_tsoutliers_signal("https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/outliers/cran_tsoutliers_ipi_France.csv", iconv_time = True)
