PYTHON=python3


HeartRateTimeSeries/HeartRateTimeSeries_series1.py  :  
	 -$(PYTHON)  tests/HeartRateTimeSeries/HeartRateTimeSeries_series1.py  >  logs/HeartRateTimeSeries_HeartRateTimeSeries_series1.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/HeartRateTimeSeries_HeartRateTimeSeries_series1.log logs/HeartRateTimeSeries_HeartRateTimeSeries_series1.log  >  logs/HeartRateTimeSeries_HeartRateTimeSeries_series1.log.diff
	 tail -10  logs/HeartRateTimeSeries_HeartRateTimeSeries_series1.log.diff 

HeartRateTimeSeries/HeartRateTimeSeries_series2.py  :  
	 -$(PYTHON)  tests/HeartRateTimeSeries/HeartRateTimeSeries_series2.py  >  logs/HeartRateTimeSeries_HeartRateTimeSeries_series2.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/HeartRateTimeSeries_HeartRateTimeSeries_series2.log logs/HeartRateTimeSeries_HeartRateTimeSeries_series2.log  >  logs/HeartRateTimeSeries_HeartRateTimeSeries_series2.log.diff
	 tail -10  logs/HeartRateTimeSeries_HeartRateTimeSeries_series2.log.diff 

HeartRateTimeSeries/HeartRateTimeSeries_series3.py  :  
	 -$(PYTHON)  tests/HeartRateTimeSeries/HeartRateTimeSeries_series3.py  >  logs/HeartRateTimeSeries_HeartRateTimeSeries_series3.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/HeartRateTimeSeries_HeartRateTimeSeries_series3.log logs/HeartRateTimeSeries_HeartRateTimeSeries_series3.log  >  logs/HeartRateTimeSeries_HeartRateTimeSeries_series3.log.diff
	 tail -10  logs/HeartRateTimeSeries_HeartRateTimeSeries_series3.log.diff 

HeartRateTimeSeries/HeartRateTimeSeries_series4.py  :  
	 -$(PYTHON)  tests/HeartRateTimeSeries/HeartRateTimeSeries_series4.py  >  logs/HeartRateTimeSeries_HeartRateTimeSeries_series4.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/HeartRateTimeSeries_HeartRateTimeSeries_series4.log logs/HeartRateTimeSeries_HeartRateTimeSeries_series4.log  >  logs/HeartRateTimeSeries_HeartRateTimeSeries_series4.log.diff
	 tail -10  logs/HeartRateTimeSeries_HeartRateTimeSeries_series4.log.diff 



 HeartRateTimeSeries :  HeartRateTimeSeries/HeartRateTimeSeries_series4.py HeartRateTimeSeries/HeartRateTimeSeries_series3.py HeartRateTimeSeries/HeartRateTimeSeries_series2.py HeartRateTimeSeries/HeartRateTimeSeries_series1.py  
 

artificial/__init__.py  :  
	 -$(PYTHON)  tests/artificial/__init__.py  >  logs/artificial___init__.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/artificial___init__.log logs/artificial___init__.log  >  logs/artificial___init__.log.diff
	 tail -10  logs/artificial___init__.log.diff 

artificial/process_artificial_dataset.py  :  
	 -$(PYTHON)  tests/artificial/process_artificial_dataset.py  >  logs/artificial_process_artificial_dataset.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/artificial_process_artificial_dataset.log logs/artificial_process_artificial_dataset.log  >  logs/artificial_process_artificial_dataset.log.diff
	 tail -10  logs/artificial_process_artificial_dataset.log.diff 



 artificial :  artificial/process_artificial_dataset.py artificial/__init__.py  
 

basic_checks/issue_46_min_max_issues.py  :  
	 -$(PYTHON)  tests/basic_checks/issue_46_min_max_issues.py  >  logs/basic_checks_issue_46_min_max_issues.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/basic_checks_issue_46_min_max_issues.log logs/basic_checks_issue_46_min_max_issues.log  >  logs/basic_checks_issue_46_min_max_issues.log.diff
	 tail -10  logs/basic_checks_issue_46_min_max_issues.log.diff 

basic_checks/issue_46_negative_horizon.py  :  
	 -$(PYTHON)  tests/basic_checks/issue_46_negative_horizon.py  >  logs/basic_checks_issue_46_negative_horizon.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/basic_checks_issue_46_negative_horizon.log logs/basic_checks_issue_46_negative_horizon.log  >  logs/basic_checks_issue_46_negative_horizon.log.diff
	 tail -10  logs/basic_checks_issue_46_negative_horizon.log.diff 

basic_checks/issue_46_non_existant_signal_column.py  :  
	 -$(PYTHON)  tests/basic_checks/issue_46_non_existant_signal_column.py  >  logs/basic_checks_issue_46_non_existant_signal_column.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/basic_checks_issue_46_non_existant_signal_column.log logs/basic_checks_issue_46_non_existant_signal_column.log  >  logs/basic_checks_issue_46_non_existant_signal_column.log.diff
	 tail -10  logs/basic_checks_issue_46_non_existant_signal_column.log.diff 

basic_checks/issue_46_non_existant_time_column.py  :  
	 -$(PYTHON)  tests/basic_checks/issue_46_non_existant_time_column.py  >  logs/basic_checks_issue_46_non_existant_time_column.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/basic_checks_issue_46_non_existant_time_column.log logs/basic_checks_issue_46_non_existant_time_column.log  >  logs/basic_checks_issue_46_non_existant_time_column.log.diff
	 tail -10  logs/basic_checks_issue_46_non_existant_time_column.log.diff 

basic_checks/issue_46_one_or_two_rows.py  :  
	 -$(PYTHON)  tests/basic_checks/issue_46_one_or_two_rows.py  >  logs/basic_checks_issue_46_one_or_two_rows.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/basic_checks_issue_46_one_or_two_rows.log logs/basic_checks_issue_46_one_or_two_rows.log  >  logs/basic_checks_issue_46_one_or_two_rows.log.diff
	 tail -10  logs/basic_checks_issue_46_one_or_two_rows.log.diff 

basic_checks/issue_46_wrong_type_date_column.py  :  
	 -$(PYTHON)  tests/basic_checks/issue_46_wrong_type_date_column.py  >  logs/basic_checks_issue_46_wrong_type_date_column.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/basic_checks_issue_46_wrong_type_date_column.log logs/basic_checks_issue_46_wrong_type_date_column.log  >  logs/basic_checks_issue_46_wrong_type_date_column.log.diff
	 tail -10  logs/basic_checks_issue_46_wrong_type_date_column.log.diff 

basic_checks/issue_46_wrong_type_signal_column.py  :  
	 -$(PYTHON)  tests/basic_checks/issue_46_wrong_type_signal_column.py  >  logs/basic_checks_issue_46_wrong_type_signal_column.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/basic_checks_issue_46_wrong_type_signal_column.log logs/basic_checks_issue_46_wrong_type_signal_column.log  >  logs/basic_checks_issue_46_wrong_type_signal_column.log.diff
	 tail -10  logs/basic_checks_issue_46_wrong_type_signal_column.log.diff 



 basic_checks :  basic_checks/issue_46_wrong_type_signal_column.py basic_checks/issue_46_wrong_type_date_column.py basic_checks/issue_46_one_or_two_rows.py basic_checks/issue_46_non_existant_time_column.py basic_checks/issue_46_non_existant_signal_column.py basic_checks/issue_46_negative_horizon.py basic_checks/issue_46_min_max_issues.py  
 

bench/test_M1Comp.py  :  
	 -$(PYTHON)  tests/bench/test_M1Comp.py  >  logs/bench_test_M1Comp.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bench_test_M1Comp.log logs/bench_test_M1Comp.log  >  logs/bench_test_M1Comp.log.diff
	 tail -10  logs/bench_test_M1Comp.log.diff 

bench/test_M2Comp.py  :  
	 -$(PYTHON)  tests/bench/test_M2Comp.py  >  logs/bench_test_M2Comp.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bench_test_M2Comp.log logs/bench_test_M2Comp.log  >  logs/bench_test_M2Comp.log.diff
	 tail -10  logs/bench_test_M2Comp.log.diff 

bench/test_M3Comp.py  :  
	 -$(PYTHON)  tests/bench/test_M3Comp.py  >  logs/bench_test_M3Comp.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bench_test_M3Comp.log logs/bench_test_M3Comp.log  >  logs/bench_test_M3Comp.log.diff
	 tail -10  logs/bench_test_M3Comp.log.diff 

bench/test_M4Comp.py  :  
	 -$(PYTHON)  tests/bench/test_M4Comp.py  >  logs/bench_test_M4Comp.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bench_test_M4Comp.log logs/bench_test_M4Comp.log  >  logs/bench_test_M4Comp.log.diff
	 tail -10  logs/bench_test_M4Comp.log.diff 

bench/test_M4_C0808_failure.py  :  
	 -$(PYTHON)  tests/bench/test_M4_C0808_failure.py  >  logs/bench_test_M4_C0808_failure.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bench_test_M4_C0808_failure.log logs/bench_test_M4_C0808_failure.log  >  logs/bench_test_M4_C0808_failure.log.diff
	 tail -10  logs/bench_test_M4_C0808_failure.log.diff 

bench/test_M4_ECON1830_failure.py  :  
	 -$(PYTHON)  tests/bench/test_M4_ECON1830_failure.py  >  logs/bench_test_M4_ECON1830_failure.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bench_test_M4_ECON1830_failure.log logs/bench_test_M4_ECON1830_failure.log  >  logs/bench_test_M4_ECON1830_failure.log.diff
	 tail -10  logs/bench_test_M4_ECON1830_failure.log.diff 

bench/test_M4_FIN27_failure.py  :  
	 -$(PYTHON)  tests/bench/test_M4_FIN27_failure.py  >  logs/bench_test_M4_FIN27_failure.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bench_test_M4_FIN27_failure.log logs/bench_test_M4_FIN27_failure.log  >  logs/bench_test_M4_FIN27_failure.log.diff
	 tail -10  logs/bench_test_M4_FIN27_failure.log.diff 

bench/test_M4_FIN28_failure.py  :  
	 -$(PYTHON)  tests/bench/test_M4_FIN28_failure.py  >  logs/bench_test_M4_FIN28_failure.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bench_test_M4_FIN28_failure.log logs/bench_test_M4_FIN28_failure.log  >  logs/bench_test_M4_FIN28_failure.log.diff
	 tail -10  logs/bench_test_M4_FIN28_failure.log.diff 

bench/test_M4_FIN29_failure.py  :  
	 -$(PYTHON)  tests/bench/test_M4_FIN29_failure.py  >  logs/bench_test_M4_FIN29_failure.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bench_test_M4_FIN29_failure.log logs/bench_test_M4_FIN29_failure.log  >  logs/bench_test_M4_FIN29_failure.log.diff
	 tail -10  logs/bench_test_M4_FIN29_failure.log.diff 

bench/test_MComp.py  :  
	 -$(PYTHON)  tests/bench/test_MComp.py  >  logs/bench_test_MComp.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bench_test_MComp.log logs/bench_test_MComp.log  >  logs/bench_test_MComp.log.diff
	 tail -10  logs/bench_test_MComp.log.diff 

bench/test_MWH_datasets.py  :  
	 -$(PYTHON)  tests/bench/test_MWH_datasets.py  >  logs/bench_test_MWH_datasets.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bench_test_MWH_datasets.log logs/bench_test_MWH_datasets.log  >  logs/bench_test_MWH_datasets.log.diff
	 tail -10  logs/bench_test_MWH_datasets.log.diff 

bench/test_NN3.py  :  
	 -$(PYTHON)  tests/bench/test_NN3.py  >  logs/bench_test_NN3.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bench_test_NN3.log logs/bench_test_NN3.log  >  logs/bench_test_NN3.log.diff
	 tail -10  logs/bench_test_NN3.log.diff 

bench/test_NN5.py  :  
	 -$(PYTHON)  tests/bench/test_NN5.py  >  logs/bench_test_NN5.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bench_test_NN5.log logs/bench_test_NN5.log  >  logs/bench_test_NN5.log.diff
	 tail -10  logs/bench_test_NN5.log.diff 

bench/test_artificial_one_dataset.py  :  
	 -$(PYTHON)  tests/bench/test_artificial_one_dataset.py  >  logs/bench_test_artificial_one_dataset.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bench_test_artificial_one_dataset.log logs/bench_test_artificial_one_dataset.log  >  logs/bench_test_artificial_one_dataset.log.diff
	 tail -10  logs/bench_test_artificial_one_dataset.log.diff 

bench/test_stock_prices.py  :  
	 -$(PYTHON)  tests/bench/test_stock_prices.py  >  logs/bench_test_stock_prices.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bench_test_stock_prices.log logs/bench_test_stock_prices.log  >  logs/bench_test_stock_prices.log.diff
	 tail -10  logs/bench_test_stock_prices.log.diff 

bench/test_yahoo.py  :  
	 -$(PYTHON)  tests/bench/test_yahoo.py  >  logs/bench_test_yahoo.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bench_test_yahoo.log logs/bench_test_yahoo.log  >  logs/bench_test_yahoo.log.diff
	 tail -10  logs/bench_test_yahoo.log.diff 

bench/test_yahoo_cac40.py  :  
	 -$(PYTHON)  tests/bench/test_yahoo_cac40.py  >  logs/bench_test_yahoo_cac40.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bench_test_yahoo_cac40.log logs/bench_test_yahoo_cac40.log  >  logs/bench_test_yahoo_cac40.log.diff
	 tail -10  logs/bench_test_yahoo_cac40.log.diff 

bench/test_yahoo_nyse_VRS.py  :  
	 -$(PYTHON)  tests/bench/test_yahoo_nyse_VRS.py  >  logs/bench_test_yahoo_nyse_VRS.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bench_test_yahoo_nyse_VRS.log logs/bench_test_yahoo_nyse_VRS.log  >  logs/bench_test_yahoo_nyse_VRS.log.diff
	 tail -10  logs/bench_test_yahoo_nyse_VRS.log.diff 



 bench :  bench/test_yahoo_nyse_VRS.py bench/test_yahoo_cac40.py bench/test_yahoo.py bench/test_stock_prices.py bench/test_artificial_one_dataset.py bench/test_NN5.py bench/test_NN3.py bench/test_MWH_datasets.py bench/test_MComp.py bench/test_M4_FIN29_failure.py bench/test_M4_FIN28_failure.py bench/test_M4_FIN27_failure.py bench/test_M4_ECON1830_failure.py bench/test_M4_C0808_failure.py bench/test_M4Comp.py bench/test_M3Comp.py bench/test_M2Comp.py bench/test_M1Comp.py  
 

bugs/test_artificial_bug_1.py  :  
	 -$(PYTHON)  tests/bugs/test_artificial_bug_1.py  >  logs/bugs_test_artificial_bug_1.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs_test_artificial_bug_1.log logs/bugs_test_artificial_bug_1.log  >  logs/bugs_test_artificial_bug_1.log.diff
	 tail -10  logs/bugs_test_artificial_bug_1.log.diff 

bugs/test_artificial_bug_2.py  :  
	 -$(PYTHON)  tests/bugs/test_artificial_bug_2.py  >  logs/bugs_test_artificial_bug_2.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs_test_artificial_bug_2.log logs/bugs_test_artificial_bug_2.log  >  logs/bugs_test_artificial_bug_2.log.diff
	 tail -10  logs/bugs_test_artificial_bug_2.log.diff 

bugs/test_random_exogenous.py  :  
	 -$(PYTHON)  tests/bugs/test_random_exogenous.py  >  logs/bugs_test_random_exogenous.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs_test_random_exogenous.log logs/bugs_test_random_exogenous.log  >  logs/bugs_test_random_exogenous.log.diff
	 tail -10  logs/bugs_test_random_exogenous.log.diff 

bugs/issue_34/issue_34.py  :  
	 -$(PYTHON)  tests/bugs/issue_34/issue_34.py  >  logs/bugs_issue_34_issue_34.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs_issue_34_issue_34.log logs/bugs_issue_34_issue_34.log  >  logs/bugs_issue_34_issue_34.log.diff
	 tail -10  logs/bugs_issue_34_issue_34.log.diff 

bugs/issue_34/issue_34_1.py  :  
	 -$(PYTHON)  tests/bugs/issue_34/issue_34_1.py  >  logs/bugs_issue_34_issue_34_1.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs_issue_34_issue_34_1.log logs/bugs_issue_34_issue_34_1.log  >  logs/bugs_issue_34_issue_34_1.log.diff
	 tail -10  logs/bugs_issue_34_issue_34_1.log.diff 

bugs/issue_34/test_artificial_1024__poly_7_12_100.py  :  
	 -$(PYTHON)  tests/bugs/issue_34/test_artificial_1024__poly_7_12_100.py  >  logs/bugs_issue_34_test_artificial_1024__poly_7_12_100.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs_issue_34_test_artificial_1024__poly_7_12_100.log logs/bugs_issue_34_test_artificial_1024__poly_7_12_100.log  >  logs/bugs_issue_34_test_artificial_1024__poly_7_12_100.log.diff
	 tail -10  logs/bugs_issue_34_test_artificial_1024__poly_7_12_100.log.diff 

bugs/issue_34/test_artificial_1024_diff_poly_0__0.py  :  
	 -$(PYTHON)  tests/bugs/issue_34/test_artificial_1024_diff_poly_0__0.py  >  logs/bugs_issue_34_test_artificial_1024_diff_poly_0__0.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs_issue_34_test_artificial_1024_diff_poly_0__0.log logs/bugs_issue_34_test_artificial_1024_diff_poly_0__0.log  >  logs/bugs_issue_34_test_artificial_1024_diff_poly_0__0.log.diff
	 tail -10  logs/bugs_issue_34_test_artificial_1024_diff_poly_0__0.log.diff 

bugs/issue_34/test_artificial_1024_diff_poly_0__100.py  :  
	 -$(PYTHON)  tests/bugs/issue_34/test_artificial_1024_diff_poly_0__100.py  >  logs/bugs_issue_34_test_artificial_1024_diff_poly_0__100.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs_issue_34_test_artificial_1024_diff_poly_0__100.log logs/bugs_issue_34_test_artificial_1024_diff_poly_0__100.log  >  logs/bugs_issue_34_test_artificial_1024_diff_poly_0__100.log.diff
	 tail -10  logs/bugs_issue_34_test_artificial_1024_diff_poly_0__100.log.diff 

bugs/issue_34/test_artificial_1024_inv_constant_30__20.py  :  
	 -$(PYTHON)  tests/bugs/issue_34/test_artificial_1024_inv_constant_30__20.py  >  logs/bugs_issue_34_test_artificial_1024_inv_constant_30__20.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs_issue_34_test_artificial_1024_inv_constant_30__20.log logs/bugs_issue_34_test_artificial_1024_inv_constant_30__20.log  >  logs/bugs_issue_34_test_artificial_1024_inv_constant_30__20.log.diff
	 tail -10  logs/bugs_issue_34_test_artificial_1024_inv_constant_30__20.log.diff 

bugs/issue_34/test_artificial_1024_inv_linear_7_12_100.py  :  
	 -$(PYTHON)  tests/bugs/issue_34/test_artificial_1024_inv_linear_7_12_100.py  >  logs/bugs_issue_34_test_artificial_1024_inv_linear_7_12_100.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs_issue_34_test_artificial_1024_inv_linear_7_12_100.log logs/bugs_issue_34_test_artificial_1024_inv_linear_7_12_100.log  >  logs/bugs_issue_34_test_artificial_1024_inv_linear_7_12_100.log.diff
	 tail -10  logs/bugs_issue_34_test_artificial_1024_inv_linear_7_12_100.log.diff 

bugs/issue_34/test_artificial_1024_log_linear_30_12_20.py  :  
	 -$(PYTHON)  tests/bugs/issue_34/test_artificial_1024_log_linear_30_12_20.py  >  logs/bugs_issue_34_test_artificial_1024_log_linear_30_12_20.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs_issue_34_test_artificial_1024_log_linear_30_12_20.log logs/bugs_issue_34_test_artificial_1024_log_linear_30_12_20.log  >  logs/bugs_issue_34_test_artificial_1024_log_linear_30_12_20.log.diff
	 tail -10  logs/bugs_issue_34_test_artificial_1024_log_linear_30_12_20.log.diff 

bugs/issue_34/test_artificial_1024_log_linear_5_12_100.py  :  
	 -$(PYTHON)  tests/bugs/issue_34/test_artificial_1024_log_linear_5_12_100.py  >  logs/bugs_issue_34_test_artificial_1024_log_linear_5_12_100.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs_issue_34_test_artificial_1024_log_linear_5_12_100.log logs/bugs_issue_34_test_artificial_1024_log_linear_5_12_100.log  >  logs/bugs_issue_34_test_artificial_1024_log_linear_5_12_100.log.diff
	 tail -10  logs/bugs_issue_34_test_artificial_1024_log_linear_5_12_100.log.diff 

bugs/issue_34/test_artificial_1024_sqr_poly_5_12_20.py  :  
	 -$(PYTHON)  tests/bugs/issue_34/test_artificial_1024_sqr_poly_5_12_20.py  >  logs/bugs_issue_34_test_artificial_1024_sqr_poly_5_12_20.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs_issue_34_test_artificial_1024_sqr_poly_5_12_20.log logs/bugs_issue_34_test_artificial_1024_sqr_poly_5_12_20.log  >  logs/bugs_issue_34_test_artificial_1024_sqr_poly_5_12_20.log.diff
	 tail -10  logs/bugs_issue_34_test_artificial_1024_sqr_poly_5_12_20.log.diff 

bugs/issue_34/test_artificial_1024_sqrt_constant_7_12_100.py  :  
	 -$(PYTHON)  tests/bugs/issue_34/test_artificial_1024_sqrt_constant_7_12_100.py  >  logs/bugs_issue_34_test_artificial_1024_sqrt_constant_7_12_100.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs_issue_34_test_artificial_1024_sqrt_constant_7_12_100.log logs/bugs_issue_34_test_artificial_1024_sqrt_constant_7_12_100.log  >  logs/bugs_issue_34_test_artificial_1024_sqrt_constant_7_12_100.log.diff
	 tail -10  logs/bugs_issue_34_test_artificial_1024_sqrt_constant_7_12_100.log.diff 

bugs/issue_34/test_artificial_1024_sqrt_linear_30_12_20.py  :  
	 -$(PYTHON)  tests/bugs/issue_34/test_artificial_1024_sqrt_linear_30_12_20.py  >  logs/bugs_issue_34_test_artificial_1024_sqrt_linear_30_12_20.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs_issue_34_test_artificial_1024_sqrt_linear_30_12_20.log logs/bugs_issue_34_test_artificial_1024_sqrt_linear_30_12_20.log  >  logs/bugs_issue_34_test_artificial_1024_sqrt_linear_30_12_20.log.diff
	 tail -10  logs/bugs_issue_34_test_artificial_1024_sqrt_linear_30_12_20.log.diff 

bugs/issue_46/issue_46_one_or_two_rows.py  :  
	 -$(PYTHON)  tests/bugs/issue_46/issue_46_one_or_two_rows.py  >  logs/bugs_issue_46_issue_46_one_or_two_rows.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs_issue_46_issue_46_one_or_two_rows.log logs/bugs_issue_46_issue_46_one_or_two_rows.log  >  logs/bugs_issue_46_issue_46_one_or_two_rows.log.diff
	 tail -10  logs/bugs_issue_46_issue_46_one_or_two_rows.log.diff 

bugs/issue_32/bug_test_ozone_heroku_1.py  :  
	 -$(PYTHON)  tests/bugs/issue_32/bug_test_ozone_heroku_1.py  >  logs/bugs_issue_32_bug_test_ozone_heroku_1.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs_issue_32_bug_test_ozone_heroku_1.log logs/bugs_issue_32_bug_test_ozone_heroku_1.log  >  logs/bugs_issue_32_bug_test_ozone_heroku_1.log.diff
	 tail -10  logs/bugs_issue_32_bug_test_ozone_heroku_1.log.diff 

bugs/issue_32/bug_test_ozone_heroku_1_ws.py  :  
	 -$(PYTHON)  tests/bugs/issue_32/bug_test_ozone_heroku_1_ws.py  >  logs/bugs_issue_32_bug_test_ozone_heroku_1_ws.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs_issue_32_bug_test_ozone_heroku_1_ws.log logs/bugs_issue_32_bug_test_ozone_heroku_1_ws.log  >  logs/bugs_issue_32_bug_test_ozone_heroku_1_ws.log.diff
	 tail -10  logs/bugs_issue_32_bug_test_ozone_heroku_1_ws.log.diff 

bugs/issue_22/test_artificial_1024_cumsum_constant_5__20.py  :  
	 -$(PYTHON)  tests/bugs/issue_22/test_artificial_1024_cumsum_constant_5__20.py  >  logs/bugs_issue_22_test_artificial_1024_cumsum_constant_5__20.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs_issue_22_test_artificial_1024_cumsum_constant_5__20.log logs/bugs_issue_22_test_artificial_1024_cumsum_constant_5__20.log  >  logs/bugs_issue_22_test_artificial_1024_cumsum_constant_5__20.log.diff
	 tail -10  logs/bugs_issue_22_test_artificial_1024_cumsum_constant_5__20.log.diff 

bugs/issue_29/test_mem_1.py  :  
	 -$(PYTHON)  tests/bugs/issue_29/test_mem_1.py  >  logs/bugs_issue_29_test_mem_1.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs_issue_29_test_mem_1.log logs/bugs_issue_29_test_mem_1.log  >  logs/bugs_issue_29_test_mem_1.log.diff
	 tail -10  logs/bugs_issue_29_test_mem_1.log.diff 

bugs/issue_36/display_version_info.py  :  
	 -$(PYTHON)  tests/bugs/issue_36/display_version_info.py  >  logs/bugs_issue_36_display_version_info.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs_issue_36_display_version_info.log logs/bugs_issue_36_display_version_info.log  >  logs/bugs_issue_36_display_version_info.log.diff
	 tail -10  logs/bugs_issue_36_display_version_info.log.diff 

bugs/issue_36/issue_36_version_info.py  :  
	 -$(PYTHON)  tests/bugs/issue_36/issue_36_version_info.py  >  logs/bugs_issue_36_issue_36_version_info.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs_issue_36_issue_36_version_info.log logs/bugs_issue_36_issue_36_version_info.log  >  logs/bugs_issue_36_issue_36_version_info.log.diff
	 tail -10  logs/bugs_issue_36_issue_36_version_info.log.diff 

bugs/issue_21/test_artificial_1024__poly_7__20.py  :  
	 -$(PYTHON)  tests/bugs/issue_21/test_artificial_1024__poly_7__20.py  >  logs/bugs_issue_21_test_artificial_1024__poly_7__20.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs_issue_21_test_artificial_1024__poly_7__20.log logs/bugs_issue_21_test_artificial_1024__poly_7__20.log  >  logs/bugs_issue_21_test_artificial_1024__poly_7__20.log.diff
	 tail -10  logs/bugs_issue_21_test_artificial_1024__poly_7__20.log.diff 

bugs/issue_21/test_ozone_max_autoreg_5.py  :  
	 -$(PYTHON)  tests/bugs/issue_21/test_ozone_max_autoreg_5.py  >  logs/bugs_issue_21_test_ozone_max_autoreg_5.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs_issue_21_test_ozone_max_autoreg_5.log logs/bugs_issue_21_test_ozone_max_autoreg_5.log  >  logs/bugs_issue_21_test_ozone_max_autoreg_5.log.diff
	 tail -10  logs/bugs_issue_21_test_ozone_max_autoreg_5.log.diff 



 bugs :  bugs/test_random_exogenous.py bugs/test_artificial_bug_2.py bugs/test_artificial_bug_1.py bugs/issue_34/test_artificial_1024_sqrt_linear_30_12_20.py bugs/issue_34/test_artificial_1024_sqrt_constant_7_12_100.py bugs/issue_34/test_artificial_1024_sqr_poly_5_12_20.py bugs/issue_34/test_artificial_1024_log_linear_5_12_100.py bugs/issue_34/test_artificial_1024_log_linear_30_12_20.py bugs/issue_34/test_artificial_1024_inv_linear_7_12_100.py bugs/issue_34/test_artificial_1024_inv_constant_30__20.py bugs/issue_34/test_artificial_1024_diff_poly_0__100.py bugs/issue_34/test_artificial_1024_diff_poly_0__0.py bugs/issue_34/test_artificial_1024__poly_7_12_100.py bugs/issue_34/issue_34_1.py bugs/issue_34/issue_34.py bugs/issue_46/issue_46_one_or_two_rows.py bugs/issue_32/bug_test_ozone_heroku_1_ws.py bugs/issue_32/bug_test_ozone_heroku_1.py bugs/issue_22/test_artificial_1024_cumsum_constant_5__20.py bugs/issue_29/test_mem_1.py bugs/issue_36/issue_36_version_info.py bugs/issue_36/display_version_info.py bugs/issue_21/test_ozone_max_autoreg_5.py bugs/issue_21/test_artificial_1024__poly_7__20.py  
 

exog/test_ozone_exogenous.py  :  
	 -$(PYTHON)  tests/exog/test_ozone_exogenous.py  >  logs/exog_test_ozone_exogenous.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/exog_test_ozone_exogenous.log logs/exog_test_ozone_exogenous.log  >  logs/exog_test_ozone_exogenous.log.diff
	 tail -10  logs/exog_test_ozone_exogenous.log.diff 

exog/test_ozone_exogenous_profile.py  :  
	 -$(PYTHON)  tests/exog/test_ozone_exogenous_profile.py  >  logs/exog_test_ozone_exogenous_profile.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/exog_test_ozone_exogenous_profile.log logs/exog_test_ozone_exogenous_profile.log  >  logs/exog_test_ozone_exogenous_profile.log.diff
	 tail -10  logs/exog_test_ozone_exogenous_profile.log.diff 

exog/test_random_exogenous.py  :  
	 -$(PYTHON)  tests/exog/test_random_exogenous.py  >  logs/exog_test_random_exogenous.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/exog_test_random_exogenous.log logs/exog_test_random_exogenous.log  >  logs/exog_test_random_exogenous.log.diff
	 tail -10  logs/exog_test_random_exogenous.log.diff 



 exog :  exog/test_random_exogenous.py exog/test_ozone_exogenous_profile.py exog/test_ozone_exogenous.py  
 

expsmooth/expsmooth_dataset_ausgdp.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_ausgdp.py  >  logs/expsmooth_expsmooth_dataset_ausgdp.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth_expsmooth_dataset_ausgdp.log logs/expsmooth_expsmooth_dataset_ausgdp.log  >  logs/expsmooth_expsmooth_dataset_ausgdp.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_ausgdp.log.diff 

expsmooth/expsmooth_dataset_bonds.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_bonds.py  >  logs/expsmooth_expsmooth_dataset_bonds.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth_expsmooth_dataset_bonds.log logs/expsmooth_expsmooth_dataset_bonds.log  >  logs/expsmooth_expsmooth_dataset_bonds.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_bonds.log.diff 

expsmooth/expsmooth_dataset_canadagas.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_canadagas.py  >  logs/expsmooth_expsmooth_dataset_canadagas.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth_expsmooth_dataset_canadagas.log logs/expsmooth_expsmooth_dataset_canadagas.log  >  logs/expsmooth_expsmooth_dataset_canadagas.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_canadagas.log.diff 

expsmooth/expsmooth_dataset_carparts.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_carparts.py  >  logs/expsmooth_expsmooth_dataset_carparts.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth_expsmooth_dataset_carparts.log logs/expsmooth_expsmooth_dataset_carparts.log  >  logs/expsmooth_expsmooth_dataset_carparts.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_carparts.log.diff 

expsmooth/expsmooth_dataset_dji.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_dji.py  >  logs/expsmooth_expsmooth_dataset_dji.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth_expsmooth_dataset_dji.log logs/expsmooth_expsmooth_dataset_dji.log  >  logs/expsmooth_expsmooth_dataset_dji.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_dji.log.diff 

expsmooth/expsmooth_dataset_djiclose.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_djiclose.py  >  logs/expsmooth_expsmooth_dataset_djiclose.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth_expsmooth_dataset_djiclose.log logs/expsmooth_expsmooth_dataset_djiclose.log  >  logs/expsmooth_expsmooth_dataset_djiclose.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_djiclose.log.diff 

expsmooth/expsmooth_dataset_enplanements.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_enplanements.py  >  logs/expsmooth_expsmooth_dataset_enplanements.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth_expsmooth_dataset_enplanements.log logs/expsmooth_expsmooth_dataset_enplanements.log  >  logs/expsmooth_expsmooth_dataset_enplanements.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_enplanements.log.diff 

expsmooth/expsmooth_dataset_fmsales.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_fmsales.py  >  logs/expsmooth_expsmooth_dataset_fmsales.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth_expsmooth_dataset_fmsales.log logs/expsmooth_expsmooth_dataset_fmsales.log  >  logs/expsmooth_expsmooth_dataset_fmsales.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_fmsales.log.diff 

expsmooth/expsmooth_dataset_freight.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_freight.py  >  logs/expsmooth_expsmooth_dataset_freight.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth_expsmooth_dataset_freight.log logs/expsmooth_expsmooth_dataset_freight.log  >  logs/expsmooth_expsmooth_dataset_freight.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_freight.log.diff 

expsmooth/expsmooth_dataset_frexport.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_frexport.py  >  logs/expsmooth_expsmooth_dataset_frexport.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth_expsmooth_dataset_frexport.log logs/expsmooth_expsmooth_dataset_frexport.log  >  logs/expsmooth_expsmooth_dataset_frexport.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_frexport.log.diff 

expsmooth/expsmooth_dataset_gasprice.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_gasprice.py  >  logs/expsmooth_expsmooth_dataset_gasprice.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth_expsmooth_dataset_gasprice.log logs/expsmooth_expsmooth_dataset_gasprice.log  >  logs/expsmooth_expsmooth_dataset_gasprice.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_gasprice.log.diff 

expsmooth/expsmooth_dataset_hospital.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_hospital.py  >  logs/expsmooth_expsmooth_dataset_hospital.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth_expsmooth_dataset_hospital.log logs/expsmooth_expsmooth_dataset_hospital.log  >  logs/expsmooth_expsmooth_dataset_hospital.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_hospital.log.diff 

expsmooth/expsmooth_dataset_jewelry.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_jewelry.py  >  logs/expsmooth_expsmooth_dataset_jewelry.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth_expsmooth_dataset_jewelry.log logs/expsmooth_expsmooth_dataset_jewelry.log  >  logs/expsmooth_expsmooth_dataset_jewelry.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_jewelry.log.diff 

expsmooth/expsmooth_dataset_mcopper.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_mcopper.py  >  logs/expsmooth_expsmooth_dataset_mcopper.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth_expsmooth_dataset_mcopper.log logs/expsmooth_expsmooth_dataset_mcopper.log  >  logs/expsmooth_expsmooth_dataset_mcopper.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_mcopper.log.diff 

expsmooth/expsmooth_dataset_msales.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_msales.py  >  logs/expsmooth_expsmooth_dataset_msales.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth_expsmooth_dataset_msales.log logs/expsmooth_expsmooth_dataset_msales.log  >  logs/expsmooth_expsmooth_dataset_msales.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_msales.log.diff 

expsmooth/expsmooth_dataset_partx.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_partx.py  >  logs/expsmooth_expsmooth_dataset_partx.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth_expsmooth_dataset_partx.log logs/expsmooth_expsmooth_dataset_partx.log  >  logs/expsmooth_expsmooth_dataset_partx.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_partx.log.diff 

expsmooth/expsmooth_dataset_test.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_test.py  >  logs/expsmooth_expsmooth_dataset_test.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth_expsmooth_dataset_test.log logs/expsmooth_expsmooth_dataset_test.log  >  logs/expsmooth_expsmooth_dataset_test.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_test.log.diff 

expsmooth/expsmooth_dataset_ukcars.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_ukcars.py  >  logs/expsmooth_expsmooth_dataset_ukcars.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth_expsmooth_dataset_ukcars.log logs/expsmooth_expsmooth_dataset_ukcars.log  >  logs/expsmooth_expsmooth_dataset_ukcars.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_ukcars.log.diff 

expsmooth/expsmooth_dataset_unemp.cci.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_unemp.cci.py  >  logs/expsmooth_expsmooth_dataset_unemp.cci.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth_expsmooth_dataset_unemp.cci.log logs/expsmooth_expsmooth_dataset_unemp.cci.log  >  logs/expsmooth_expsmooth_dataset_unemp.cci.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_unemp.cci.log.diff 

expsmooth/expsmooth_dataset_usgdp.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_usgdp.py  >  logs/expsmooth_expsmooth_dataset_usgdp.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth_expsmooth_dataset_usgdp.log logs/expsmooth_expsmooth_dataset_usgdp.log  >  logs/expsmooth_expsmooth_dataset_usgdp.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_usgdp.log.diff 

expsmooth/expsmooth_dataset_usnetelec.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_usnetelec.py  >  logs/expsmooth_expsmooth_dataset_usnetelec.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth_expsmooth_dataset_usnetelec.log logs/expsmooth_expsmooth_dataset_usnetelec.log  >  logs/expsmooth_expsmooth_dataset_usnetelec.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_usnetelec.log.diff 

expsmooth/expsmooth_dataset_utility.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_utility.py  >  logs/expsmooth_expsmooth_dataset_utility.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth_expsmooth_dataset_utility.log logs/expsmooth_expsmooth_dataset_utility.log  >  logs/expsmooth_expsmooth_dataset_utility.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_utility.log.diff 

expsmooth/expsmooth_dataset_vehicles.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_vehicles.py  >  logs/expsmooth_expsmooth_dataset_vehicles.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth_expsmooth_dataset_vehicles.log logs/expsmooth_expsmooth_dataset_vehicles.log  >  logs/expsmooth_expsmooth_dataset_vehicles.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_vehicles.log.diff 

expsmooth/expsmooth_dataset_visitors.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_visitors.py  >  logs/expsmooth_expsmooth_dataset_visitors.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth_expsmooth_dataset_visitors.log logs/expsmooth_expsmooth_dataset_visitors.log  >  logs/expsmooth_expsmooth_dataset_visitors.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_visitors.log.diff 

expsmooth/expsmooth_dataset_xrates.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_xrates.py  >  logs/expsmooth_expsmooth_dataset_xrates.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth_expsmooth_dataset_xrates.log logs/expsmooth_expsmooth_dataset_xrates.log  >  logs/expsmooth_expsmooth_dataset_xrates.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_xrates.log.diff 



 expsmooth :  expsmooth/expsmooth_dataset_xrates.py expsmooth/expsmooth_dataset_visitors.py expsmooth/expsmooth_dataset_vehicles.py expsmooth/expsmooth_dataset_utility.py expsmooth/expsmooth_dataset_usnetelec.py expsmooth/expsmooth_dataset_usgdp.py expsmooth/expsmooth_dataset_unemp.cci.py expsmooth/expsmooth_dataset_ukcars.py expsmooth/expsmooth_dataset_test.py expsmooth/expsmooth_dataset_partx.py expsmooth/expsmooth_dataset_msales.py expsmooth/expsmooth_dataset_mcopper.py expsmooth/expsmooth_dataset_jewelry.py expsmooth/expsmooth_dataset_hospital.py expsmooth/expsmooth_dataset_gasprice.py expsmooth/expsmooth_dataset_frexport.py expsmooth/expsmooth_dataset_freight.py expsmooth/expsmooth_dataset_fmsales.py expsmooth/expsmooth_dataset_enplanements.py expsmooth/expsmooth_dataset_djiclose.py expsmooth/expsmooth_dataset_dji.py expsmooth/expsmooth_dataset_carparts.py expsmooth/expsmooth_dataset_canadagas.py expsmooth/expsmooth_dataset_bonds.py expsmooth/expsmooth_dataset_ausgdp.py  
 

func/test_air_passengers.py  :  
	 -$(PYTHON)  tests/func/test_air_passengers.py  >  logs/func_test_air_passengers.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/func_test_air_passengers.log logs/func_test_air_passengers.log  >  logs/func_test_air_passengers.log.diff
	 tail -10  logs/func_test_air_passengers.log.diff 

func/test_ar.py  :  
	 -$(PYTHON)  tests/func/test_ar.py  >  logs/func_test_ar.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/func_test_ar.log logs/func_test_ar.log  >  logs/func_test_ar.log.diff
	 tail -10  logs/func_test_ar.log.diff 

func/test_const_signal.py  :  
	 -$(PYTHON)  tests/func/test_const_signal.py  >  logs/func_test_const_signal.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/func_test_const_signal.log logs/func_test_const_signal.log  >  logs/func_test_const_signal.log.diff
	 tail -10  logs/func_test_const_signal.log.diff 

func/test_cycle.py  :  
	 -$(PYTHON)  tests/func/test_cycle.py  >  logs/func_test_cycle.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/func_test_cycle.log logs/func_test_cycle.log  >  logs/func_test_cycle.log.diff
	 tail -10  logs/func_test_cycle.log.diff 

func/test_cycles_full.py  :  
	 -$(PYTHON)  tests/func/test_cycles_full.py  >  logs/func_test_cycles_full.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/func_test_cycles_full.log logs/func_test_cycles_full.log  >  logs/func_test_cycles_full.log.diff
	 tail -10  logs/func_test_cycles_full.log.diff 

func/test_ozone.py  :  
	 -$(PYTHON)  tests/func/test_ozone.py  >  logs/func_test_ozone.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/func_test_ozone.log logs/func_test_ozone.log  >  logs/func_test_ozone.log.diff
	 tail -10  logs/func_test_ozone.log.diff 

func/test_ozone_with_logging.py  :  
	 -$(PYTHON)  tests/func/test_ozone_with_logging.py  >  logs/func_test_ozone_with_logging.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/func_test_ozone_with_logging.log logs/func_test_ozone_with_logging.log  >  logs/func_test_ozone_with_logging.log.diff
	 tail -10  logs/func_test_ozone_with_logging.log.diff 



 func :  func/test_ozone_with_logging.py func/test_ozone.py func/test_cycles_full.py func/test_cycle.py func/test_const_signal.py func/test_ar.py func/test_air_passengers.py  
 

heroku/build_generic_heroku_model.py  :  
	 -$(PYTHON)  tests/heroku/build_generic_heroku_model.py  >  logs/heroku_build_generic_heroku_model.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/heroku_build_generic_heroku_model.log logs/heroku_build_generic_heroku_model.log  >  logs/heroku_build_generic_heroku_model.log.diff
	 tail -10  logs/heroku_build_generic_heroku_model.log.diff 

heroku/test_air_passengers_heroku.py  :  
	 -$(PYTHON)  tests/heroku/test_air_passengers_heroku.py  >  logs/heroku_test_air_passengers_heroku.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/heroku_test_air_passengers_heroku.log logs/heroku_test_air_passengers_heroku.log  >  logs/heroku_test_air_passengers_heroku.log.diff
	 tail -10  logs/heroku_test_air_passengers_heroku.log.diff 

heroku/test_ozone_exog_heroku.py  :  
	 -$(PYTHON)  tests/heroku/test_ozone_exog_heroku.py  >  logs/heroku_test_ozone_exog_heroku.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/heroku_test_ozone_exog_heroku.log logs/heroku_test_ozone_exog_heroku.log  >  logs/heroku_test_ozone_exog_heroku.log.diff
	 tail -10  logs/heroku_test_ozone_exog_heroku.log.diff 

heroku/test_ozone_heroku.py  :  
	 -$(PYTHON)  tests/heroku/test_ozone_heroku.py  >  logs/heroku_test_ozone_heroku.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/heroku_test_ozone_heroku.log logs/heroku_test_ozone_heroku.log  >  logs/heroku_test_ozone_heroku.log.diff
	 tail -10  logs/heroku_test_ozone_heroku.log.diff 

heroku/test_seasonal_week_of_year.py  :  
	 -$(PYTHON)  tests/heroku/test_seasonal_week_of_year.py  >  logs/heroku_test_seasonal_week_of_year.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/heroku_test_seasonal_week_of_year.log logs/heroku_test_seasonal_week_of_year.log  >  logs/heroku_test_seasonal_week_of_year.log.diff
	 tail -10  logs/heroku_test_seasonal_week_of_year.log.diff 



 heroku :  heroku/test_seasonal_week_of_year.py heroku/test_ozone_heroku.py heroku/test_ozone_exog_heroku.py heroku/test_air_passengers_heroku.py heroku/build_generic_heroku_model.py  
 

hierarchical/test_grouped.py  :  
	 -$(PYTHON)  tests/hierarchical/test_grouped.py  >  logs/hierarchical_test_grouped.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical_test_grouped.log logs/hierarchical_test_grouped.log  >  logs/hierarchical_test_grouped.log.diff
	 tail -10  logs/hierarchical_test_grouped.log.diff 

hierarchical/test_grouped_signals_AllMethods.py  :  
	 -$(PYTHON)  tests/hierarchical/test_grouped_signals_AllMethods.py  >  logs/hierarchical_test_grouped_signals_AllMethods.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical_test_grouped_signals_AllMethods.log logs/hierarchical_test_grouped_signals_AllMethods.log  >  logs/hierarchical_test_grouped_signals_AllMethods.log.diff
	 tail -10  logs/hierarchical_test_grouped_signals_AllMethods.log.diff 

hierarchical/test_grouped_signals_BU.py  :  
	 -$(PYTHON)  tests/hierarchical/test_grouped_signals_BU.py  >  logs/hierarchical_test_grouped_signals_BU.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical_test_grouped_signals_BU.log logs/hierarchical_test_grouped_signals_BU.log  >  logs/hierarchical_test_grouped_signals_BU.log.diff
	 tail -10  logs/hierarchical_test_grouped_signals_BU.log.diff 

hierarchical/test_grouped_signals_MO.py  :  
	 -$(PYTHON)  tests/hierarchical/test_grouped_signals_MO.py  >  logs/hierarchical_test_grouped_signals_MO.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical_test_grouped_signals_MO.log logs/hierarchical_test_grouped_signals_MO.log  >  logs/hierarchical_test_grouped_signals_MO.log.diff
	 tail -10  logs/hierarchical_test_grouped_signals_MO.log.diff 

hierarchical/test_grouped_signals_OC.py  :  
	 -$(PYTHON)  tests/hierarchical/test_grouped_signals_OC.py  >  logs/hierarchical_test_grouped_signals_OC.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical_test_grouped_signals_OC.log logs/hierarchical_test_grouped_signals_OC.log  >  logs/hierarchical_test_grouped_signals_OC.log.diff
	 tail -10  logs/hierarchical_test_grouped_signals_OC.log.diff 

hierarchical/test_grouped_signals_TD.py  :  
	 -$(PYTHON)  tests/hierarchical/test_grouped_signals_TD.py  >  logs/hierarchical_test_grouped_signals_TD.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical_test_grouped_signals_TD.log logs/hierarchical_test_grouped_signals_TD.log  >  logs/hierarchical_test_grouped_signals_TD.log.diff
	 tail -10  logs/hierarchical_test_grouped_signals_TD.log.diff 

hierarchical/test_hierarchy_AU.py  :  
	 -$(PYTHON)  tests/hierarchical/test_hierarchy_AU.py  >  logs/hierarchical_test_hierarchy_AU.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical_test_hierarchy_AU.log logs/hierarchical_test_hierarchy_AU.log  >  logs/hierarchical_test_hierarchy_AU.log.diff
	 tail -10  logs/hierarchical_test_hierarchy_AU.log.diff 

hierarchical/test_hierarchy_AU_AllMethods.py  :  
	 -$(PYTHON)  tests/hierarchical/test_hierarchy_AU_AllMethods.py  >  logs/hierarchical_test_hierarchy_AU_AllMethods.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical_test_hierarchy_AU_AllMethods.log logs/hierarchical_test_hierarchy_AU_AllMethods.log  >  logs/hierarchical_test_hierarchy_AU_AllMethods.log.diff
	 tail -10  logs/hierarchical_test_hierarchy_AU_AllMethods.log.diff 

hierarchical/test_hierarchy_AU_BU.py  :  
	 -$(PYTHON)  tests/hierarchical/test_hierarchy_AU_BU.py  >  logs/hierarchical_test_hierarchy_AU_BU.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical_test_hierarchy_AU_BU.log logs/hierarchical_test_hierarchy_AU_BU.log  >  logs/hierarchical_test_hierarchy_AU_BU.log.diff
	 tail -10  logs/hierarchical_test_hierarchy_AU_BU.log.diff 

hierarchical/test_hierarchy_AU_MO.py  :  
	 -$(PYTHON)  tests/hierarchical/test_hierarchy_AU_MO.py  >  logs/hierarchical_test_hierarchy_AU_MO.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical_test_hierarchy_AU_MO.log logs/hierarchical_test_hierarchy_AU_MO.log  >  logs/hierarchical_test_hierarchy_AU_MO.log.diff
	 tail -10  logs/hierarchical_test_hierarchy_AU_MO.log.diff 

hierarchical/test_hierarchy_AU_OC.py  :  
	 -$(PYTHON)  tests/hierarchical/test_hierarchy_AU_OC.py  >  logs/hierarchical_test_hierarchy_AU_OC.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical_test_hierarchy_AU_OC.log logs/hierarchical_test_hierarchy_AU_OC.log  >  logs/hierarchical_test_hierarchy_AU_OC.log.diff
	 tail -10  logs/hierarchical_test_hierarchy_AU_OC.log.diff 

hierarchical/test_hierarchy_AU_TD.py  :  
	 -$(PYTHON)  tests/hierarchical/test_hierarchy_AU_TD.py  >  logs/hierarchical_test_hierarchy_AU_TD.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical_test_hierarchy_AU_TD.log logs/hierarchical_test_hierarchy_AU_TD.log  >  logs/hierarchical_test_hierarchy_AU_TD.log.diff
	 tail -10  logs/hierarchical_test_hierarchy_AU_TD.log.diff 



 hierarchical :  hierarchical/test_hierarchy_AU_TD.py hierarchical/test_hierarchy_AU_OC.py hierarchical/test_hierarchy_AU_MO.py hierarchical/test_hierarchy_AU_BU.py hierarchical/test_hierarchy_AU_AllMethods.py hierarchical/test_hierarchy_AU.py hierarchical/test_grouped_signals_TD.py hierarchical/test_grouped_signals_OC.py hierarchical/test_grouped_signals_MO.py hierarchical/test_grouped_signals_BU.py hierarchical/test_grouped_signals_AllMethods.py hierarchical/test_grouped.py  
 

model_control/test_ozone_all_models_enabled.py  :  
	 -$(PYTHON)  tests/model_control/test_ozone_all_models_enabled.py  >  logs/model_control_test_ozone_all_models_enabled.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/model_control_test_ozone_all_models_enabled.log logs/model_control_test_ozone_all_models_enabled.log  >  logs/model_control_test_ozone_all_models_enabled.log.diff
	 tail -10  logs/model_control_test_ozone_all_models_enabled.log.diff 

model_control/test_ozone_custom_models_enabled.py  :  
	 -$(PYTHON)  tests/model_control/test_ozone_custom_models_enabled.py  >  logs/model_control_test_ozone_custom_models_enabled.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/model_control_test_ozone_custom_models_enabled.log logs/model_control_test_ozone_custom_models_enabled.log  >  logs/model_control_test_ozone_custom_models_enabled.log.diff
	 tail -10  logs/model_control_test_ozone_custom_models_enabled.log.diff 

model_control/test_ozone_no_models_enabled.py  :  
	 -$(PYTHON)  tests/model_control/test_ozone_no_models_enabled.py  >  logs/model_control_test_ozone_no_models_enabled.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/model_control_test_ozone_no_models_enabled.log logs/model_control_test_ozone_no_models_enabled.log  >  logs/model_control_test_ozone_no_models_enabled.log.diff
	 tail -10  logs/model_control_test_ozone_no_models_enabled.log.diff 



 model_control :  model_control/test_ozone_no_models_enabled.py model_control/test_ozone_custom_models_enabled.py model_control/test_ozone_all_models_enabled.py  
 

neuralnet/test_air_passengers_GPU.py  :  
	 -$(PYTHON)  tests/neuralnet/test_air_passengers_GPU.py  >  logs/neuralnet_test_air_passengers_GPU.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/neuralnet_test_air_passengers_GPU.log logs/neuralnet_test_air_passengers_GPU.log  >  logs/neuralnet_test_air_passengers_GPU.log.diff
	 tail -10  logs/neuralnet_test_air_passengers_GPU.log.diff 

neuralnet/test_air_passengers_rnn_only.py  :  
	 -$(PYTHON)  tests/neuralnet/test_air_passengers_rnn_only.py  >  logs/neuralnet_test_air_passengers_rnn_only.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/neuralnet_test_air_passengers_rnn_only.log logs/neuralnet_test_air_passengers_rnn_only.log  >  logs/neuralnet_test_air_passengers_rnn_only.log.diff
	 tail -10  logs/neuralnet_test_air_passengers_rnn_only.log.diff 

neuralnet/test_air_passengers_tensorflow.py  :  
	 -$(PYTHON)  tests/neuralnet/test_air_passengers_tensorflow.py  >  logs/neuralnet_test_air_passengers_tensorflow.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/neuralnet_test_air_passengers_tensorflow.log logs/neuralnet_test_air_passengers_tensorflow.log  >  logs/neuralnet_test_air_passengers_tensorflow.log.diff
	 tail -10  logs/neuralnet_test_air_passengers_tensorflow.log.diff 

neuralnet/test_ozone__GPU.py  :  
	 -$(PYTHON)  tests/neuralnet/test_ozone__GPU.py  >  logs/neuralnet_test_ozone__GPU.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/neuralnet_test_ozone__GPU.log logs/neuralnet_test_ozone__GPU.log  >  logs/neuralnet_test_ozone__GPU.log.diff
	 tail -10  logs/neuralnet_test_ozone__GPU.log.diff 

neuralnet/test_ozone_rnn_only.py  :  
	 -$(PYTHON)  tests/neuralnet/test_ozone_rnn_only.py  >  logs/neuralnet_test_ozone_rnn_only.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/neuralnet_test_ozone_rnn_only.log logs/neuralnet_test_ozone_rnn_only.log  >  logs/neuralnet_test_ozone_rnn_only.log.diff
	 tail -10  logs/neuralnet_test_ozone_rnn_only.log.diff 

neuralnet/test_ozone_tensorflow.py  :  
	 -$(PYTHON)  tests/neuralnet/test_ozone_tensorflow.py  >  logs/neuralnet_test_ozone_tensorflow.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/neuralnet_test_ozone_tensorflow.log logs/neuralnet_test_ozone_tensorflow.log  >  logs/neuralnet_test_ozone_tensorflow.log.diff
	 tail -10  logs/neuralnet_test_ozone_tensorflow.log.diff 



 neuralnet :  neuralnet/test_ozone_tensorflow.py neuralnet/test_ozone_rnn_only.py neuralnet/test_ozone__GPU.py neuralnet/test_air_passengers_tensorflow.py neuralnet/test_air_passengers_rnn_only.py neuralnet/test_air_passengers_GPU.py  
 

perf/test_cycles_full_long.py  :  
	 -$(PYTHON)  tests/perf/test_cycles_full_long.py  >  logs/perf_test_cycles_full_long.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf_test_cycles_full_long.log logs/perf_test_cycles_full_long.log  >  logs/perf_test_cycles_full_long.log.diff
	 tail -10  logs/perf_test_cycles_full_long.log.diff 

perf/test_cycles_full_long_long.py  :  
	 -$(PYTHON)  tests/perf/test_cycles_full_long_long.py  >  logs/perf_test_cycles_full_long_long.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf_test_cycles_full_long_long.log logs/perf_test_cycles_full_long_long.log  >  logs/perf_test_cycles_full_long_long.log.diff
	 tail -10  logs/perf_test_cycles_full_long_long.log.diff 

perf/test_cycles_full_long_long_2.py  :  
	 -$(PYTHON)  tests/perf/test_cycles_full_long_long_2.py  >  logs/perf_test_cycles_full_long_long_2.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf_test_cycles_full_long_long_2.log logs/perf_test_cycles_full_long_long_2.log  >  logs/perf_test_cycles_full_long_long_2.log.diff
	 tail -10  logs/perf_test_cycles_full_long_long_2.log.diff 

perf/test_ozone_debug_perf.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_debug_perf.py  >  logs/perf_test_ozone_debug_perf.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf_test_ozone_debug_perf.log logs/perf_test_ozone_debug_perf.log  >  logs/perf_test_ozone_debug_perf.log.diff
	 tail -10  logs/perf_test_ozone_debug_perf.log.diff 

perf/test_ozone_long_series.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_long_series.py  >  logs/perf_test_ozone_long_series.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf_test_ozone_long_series.log logs/perf_test_ozone_long_series.log  >  logs/perf_test_ozone_long_series.log.diff
	 tail -10  logs/perf_test_ozone_long_series.log.diff 

perf/test_ozone_long_series_2.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_long_series_2.py  >  logs/perf_test_ozone_long_series_2.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf_test_ozone_long_series_2.log logs/perf_test_ozone_long_series_2.log  >  logs/perf_test_ozone_long_series_2.log.diff
	 tail -10  logs/perf_test_ozone_long_series_2.log.diff 

perf/test_parallel.py  :  
	 -$(PYTHON)  tests/perf/test_parallel.py  >  logs/perf_test_parallel.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf_test_parallel.log logs/perf_test_parallel.log  >  logs/perf_test_parallel.log.diff
	 tail -10  logs/perf_test_parallel.log.diff 

perf/test_perf1.py  :  
	 -$(PYTHON)  tests/perf/test_perf1.py  >  logs/perf_test_perf1.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf_test_perf1.log logs/perf_test_perf1.log  >  logs/perf_test_perf1.log.diff
	 tail -10  logs/perf_test_perf1.log.diff 

perf/test_perf2.py  :  
	 -$(PYTHON)  tests/perf/test_perf2.py  >  logs/perf_test_perf2.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf_test_perf2.log logs/perf_test_perf2.log  >  logs/perf_test_perf2.log.diff
	 tail -10  logs/perf_test_perf2.log.diff 



 perf :  perf/test_perf2.py perf/test_perf1.py perf/test_parallel.py perf/test_ozone_long_series_2.py perf/test_ozone_long_series.py perf/test_ozone_debug_perf.py perf/test_cycles_full_long_long_2.py perf/test_cycles_full_long_long.py perf/test_cycles_full_long.py  
 

perfs/test_ozone_perf_measure.py  :  
	 -$(PYTHON)  tests/perfs/test_ozone_perf_measure.py  >  logs/perfs_test_ozone_perf_measure.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perfs_test_ozone_perf_measure.log logs/perfs_test_ozone_perf_measure.log  >  logs/perfs_test_ozone_perf_measure.log.diff
	 tail -10  logs/perfs_test_ozone_perf_measure.log.diff 

perfs/test_ozone_perf_measure_L1.py  :  
	 -$(PYTHON)  tests/perfs/test_ozone_perf_measure_L1.py  >  logs/perfs_test_ozone_perf_measure_L1.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perfs_test_ozone_perf_measure_L1.log logs/perfs_test_ozone_perf_measure_L1.log  >  logs/perfs_test_ozone_perf_measure_L1.log.diff
	 tail -10  logs/perfs_test_ozone_perf_measure_L1.log.diff 

perfs/test_ozone_perf_measure_L2.py  :  
	 -$(PYTHON)  tests/perfs/test_ozone_perf_measure_L2.py  >  logs/perfs_test_ozone_perf_measure_L2.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perfs_test_ozone_perf_measure_L2.log logs/perfs_test_ozone_perf_measure_L2.log  >  logs/perfs_test_ozone_perf_measure_L2.log.diff
	 tail -10  logs/perfs_test_ozone_perf_measure_L2.log.diff 

perfs/test_ozone_perf_measure_MAPE.py  :  
	 -$(PYTHON)  tests/perfs/test_ozone_perf_measure_MAPE.py  >  logs/perfs_test_ozone_perf_measure_MAPE.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perfs_test_ozone_perf_measure_MAPE.log logs/perfs_test_ozone_perf_measure_MAPE.log  >  logs/perfs_test_ozone_perf_measure_MAPE.log.diff
	 tail -10  logs/perfs_test_ozone_perf_measure_MAPE.log.diff 

perfs/test_ozone_perf_measure_MASE.py  :  
	 -$(PYTHON)  tests/perfs/test_ozone_perf_measure_MASE.py  >  logs/perfs_test_ozone_perf_measure_MASE.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perfs_test_ozone_perf_measure_MASE.log logs/perfs_test_ozone_perf_measure_MASE.log  >  logs/perfs_test_ozone_perf_measure_MASE.log.diff
	 tail -10  logs/perfs_test_ozone_perf_measure_MASE.log.diff 



 perfs :  perfs/test_ozone_perf_measure_MASE.py perfs/test_ozone_perf_measure_MAPE.py perfs/test_ozone_perf_measure_L2.py perfs/test_ozone_perf_measure_L1.py perfs/test_ozone_perf_measure.py  
 

real-life/test_sof_example.py  :  
	 -$(PYTHON)  tests/real-life/test_sof_example.py  >  logs/real-life_test_sof_example.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/real-life_test_sof_example.log logs/real-life_test_sof_example.log  >  logs/real-life_test_sof_example.log.diff
	 tail -10  logs/real-life_test_sof_example.log.diff 

real-life/test_sof_example2.py  :  
	 -$(PYTHON)  tests/real-life/test_sof_example2.py  >  logs/real-life_test_sof_example2.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/real-life_test_sof_example2.log logs/real-life_test_sof_example2.log  >  logs/real-life_test_sof_example2.log.diff
	 tail -10  logs/real-life_test_sof_example2.log.diff 



 real-life :  real-life/test_sof_example2.py real-life/test_sof_example.py  
 

svr/test_air_passengers_svr.py  :  
	 -$(PYTHON)  tests/svr/test_air_passengers_svr.py  >  logs/svr_test_air_passengers_svr.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/svr_test_air_passengers_svr.log logs/svr_test_air_passengers_svr.log  >  logs/svr_test_air_passengers_svr.log.diff
	 tail -10  logs/svr_test_air_passengers_svr.log.diff 

svr/test_air_passengers_svr_only.py  :  
	 -$(PYTHON)  tests/svr/test_air_passengers_svr_only.py  >  logs/svr_test_air_passengers_svr_only.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/svr_test_air_passengers_svr_only.log logs/svr_test_air_passengers_svr_only.log  >  logs/svr_test_air_passengers_svr_only.log.diff
	 tail -10  logs/svr_test_air_passengers_svr_only.log.diff 

svr/test_ozone_svr.py  :  
	 -$(PYTHON)  tests/svr/test_ozone_svr.py  >  logs/svr_test_ozone_svr.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/svr_test_ozone_svr.log logs/svr_test_ozone_svr.log  >  logs/svr_test_ozone_svr.log.diff
	 tail -10  logs/svr_test_ozone_svr.log.diff 

svr/test_ozone_svr_only.py  :  
	 -$(PYTHON)  tests/svr/test_ozone_svr_only.py  >  logs/svr_test_ozone_svr_only.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/svr_test_ozone_svr_only.log logs/svr_test_ozone_svr_only.log  >  logs/svr_test_ozone_svr_only.log.diff
	 tail -10  logs/svr_test_ozone_svr_only.log.diff 



 svr :  svr/test_ozone_svr_only.py svr/test_ozone_svr.py svr/test_air_passengers_svr_only.py svr/test_air_passengers_svr.py  
 

time_res/test_ozone_Daily.py  :  
	 -$(PYTHON)  tests/time_res/test_ozone_Daily.py  >  logs/time_res_test_ozone_Daily.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/time_res_test_ozone_Daily.log logs/time_res_test_ozone_Daily.log  >  logs/time_res_test_ozone_Daily.log.diff
	 tail -10  logs/time_res_test_ozone_Daily.log.diff 

time_res/test_ozone_Hourly.py  :  
	 -$(PYTHON)  tests/time_res/test_ozone_Hourly.py  >  logs/time_res_test_ozone_Hourly.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/time_res_test_ozone_Hourly.log logs/time_res_test_ozone_Hourly.log  >  logs/time_res_test_ozone_Hourly.log.diff
	 tail -10  logs/time_res_test_ozone_Hourly.log.diff 

time_res/test_ozone_Minutely.py  :  
	 -$(PYTHON)  tests/time_res/test_ozone_Minutely.py  >  logs/time_res_test_ozone_Minutely.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/time_res_test_ozone_Minutely.log logs/time_res_test_ozone_Minutely.log  >  logs/time_res_test_ozone_Minutely.log.diff
	 tail -10  logs/time_res_test_ozone_Minutely.log.diff 

time_res/test_ozone_Secondly.py  :  
	 -$(PYTHON)  tests/time_res/test_ozone_Secondly.py  >  logs/time_res_test_ozone_Secondly.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/time_res_test_ozone_Secondly.log logs/time_res_test_ozone_Secondly.log  >  logs/time_res_test_ozone_Secondly.log.diff
	 tail -10  logs/time_res_test_ozone_Secondly.log.diff 

time_res/test_ozone_Weekly.py  :  
	 -$(PYTHON)  tests/time_res/test_ozone_Weekly.py  >  logs/time_res_test_ozone_Weekly.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/time_res_test_ozone_Weekly.log logs/time_res_test_ozone_Weekly.log  >  logs/time_res_test_ozone_Weekly.log.diff
	 tail -10  logs/time_res_test_ozone_Weekly.log.diff 



 time_res :  time_res/test_ozone_Weekly.py time_res/test_ozone_Secondly.py time_res/test_ozone_Minutely.py time_res/test_ozone_Hourly.py time_res/test_ozone_Daily.py  
 

transformations/test_ozone_transf_anscombe.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_anscombe.py  >  logs/transformations_test_ozone_transf_anscombe.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/transformations_test_ozone_transf_anscombe.log logs/transformations_test_ozone_transf_anscombe.log  >  logs/transformations_test_ozone_transf_anscombe.log.diff
	 tail -10  logs/transformations_test_ozone_transf_anscombe.log.diff 

transformations/test_ozone_transf_boxcox.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_boxcox.py  >  logs/transformations_test_ozone_transf_boxcox.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/transformations_test_ozone_transf_boxcox.log logs/transformations_test_ozone_transf_boxcox.log  >  logs/transformations_test_ozone_transf_boxcox.log.diff
	 tail -10  logs/transformations_test_ozone_transf_boxcox.log.diff 

transformations/test_ozone_transf_cumsum.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_cumsum.py  >  logs/transformations_test_ozone_transf_cumsum.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/transformations_test_ozone_transf_cumsum.log logs/transformations_test_ozone_transf_cumsum.log  >  logs/transformations_test_ozone_transf_cumsum.log.diff
	 tail -10  logs/transformations_test_ozone_transf_cumsum.log.diff 

transformations/test_ozone_transf_difference.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_difference.py  >  logs/transformations_test_ozone_transf_difference.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/transformations_test_ozone_transf_difference.log logs/transformations_test_ozone_transf_difference.log  >  logs/transformations_test_ozone_transf_difference.log.diff
	 tail -10  logs/transformations_test_ozone_transf_difference.log.diff 

transformations/test_ozone_transf_fisher.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_fisher.py  >  logs/transformations_test_ozone_transf_fisher.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/transformations_test_ozone_transf_fisher.log logs/transformations_test_ozone_transf_fisher.log  >  logs/transformations_test_ozone_transf_fisher.log.diff
	 tail -10  logs/transformations_test_ozone_transf_fisher.log.diff 

transformations/test_ozone_transf_generic.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_generic.py  >  logs/transformations_test_ozone_transf_generic.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/transformations_test_ozone_transf_generic.log logs/transformations_test_ozone_transf_generic.log  >  logs/transformations_test_ozone_transf_generic.log.diff
	 tail -10  logs/transformations_test_ozone_transf_generic.log.diff 

transformations/test_ozone_transf_logit.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_logit.py  >  logs/transformations_test_ozone_transf_logit.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/transformations_test_ozone_transf_logit.log logs/transformations_test_ozone_transf_logit.log  >  logs/transformations_test_ozone_transf_logit.log.diff
	 tail -10  logs/transformations_test_ozone_transf_logit.log.diff 

transformations/test_ozone_transf_none.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_none.py  >  logs/transformations_test_ozone_transf_none.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/transformations_test_ozone_transf_none.log logs/transformations_test_ozone_transf_none.log  >  logs/transformations_test_ozone_transf_none.log.diff
	 tail -10  logs/transformations_test_ozone_transf_none.log.diff 

transformations/test_ozone_transf_quantization.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_quantization.py  >  logs/transformations_test_ozone_transf_quantization.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/transformations_test_ozone_transf_quantization.log logs/transformations_test_ozone_transf_quantization.log  >  logs/transformations_test_ozone_transf_quantization.log.diff
	 tail -10  logs/transformations_test_ozone_transf_quantization.log.diff 

transformations/test_ozone_transf_relative_difference.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_relative_difference.py  >  logs/transformations_test_ozone_transf_relative_difference.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/transformations_test_ozone_transf_relative_difference.log logs/transformations_test_ozone_transf_relative_difference.log  >  logs/transformations_test_ozone_transf_relative_difference.log.diff
	 tail -10  logs/transformations_test_ozone_transf_relative_difference.log.diff 

transformations/test_ozone_transf_relative_difference_1.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_relative_difference_1.py  >  logs/transformations_test_ozone_transf_relative_difference_1.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/transformations_test_ozone_transf_relative_difference_1.log logs/transformations_test_ozone_transf_relative_difference_1.log  >  logs/transformations_test_ozone_transf_relative_difference_1.log.diff
	 tail -10  logs/transformations_test_ozone_transf_relative_difference_1.log.diff 



 transformations :  transformations/test_ozone_transf_relative_difference_1.py transformations/test_ozone_transf_relative_difference.py transformations/test_ozone_transf_quantization.py transformations/test_ozone_transf_none.py transformations/test_ozone_transf_logit.py transformations/test_ozone_transf_generic.py transformations/test_ozone_transf_fisher.py transformations/test_ozone_transf_difference.py transformations/test_ozone_transf_cumsum.py transformations/test_ozone_transf_boxcox.py transformations/test_ozone_transf_anscombe.py  
 


# ********************************************** 

all:  artificial basic_checks bugs exog expsmooth HeartRateTimeSeries heroku hierarchical model_control perf svr transformations bench func neuralnet real-life  time_res perfs 
	

build-test :  basic_checks exog heroku hierarchical model_control svr transformations func real-life  time_res perfs 
	

