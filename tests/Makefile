PYTHON=timeout 480 python


HeartRateTimeSeries/HeartRateTimeSeries_series1.py  :  
	 -$(PYTHON)  tests/HeartRateTimeSeries/HeartRateTimeSeries_series1.py  >  logs/HeartRateTimeSeries_HeartRateTimeSeries_series1.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/HeartRateTimeSeries/HeartRateTimeSeries_series1.log logs/HeartRateTimeSeries_HeartRateTimeSeries_series1.log  >  logs/HeartRateTimeSeries_HeartRateTimeSeries_series1.log.diff
	 tail -10  logs/HeartRateTimeSeries_HeartRateTimeSeries_series1.log.diff 

HeartRateTimeSeries/HeartRateTimeSeries_series2.py  :  
	 -$(PYTHON)  tests/HeartRateTimeSeries/HeartRateTimeSeries_series2.py  >  logs/HeartRateTimeSeries_HeartRateTimeSeries_series2.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/HeartRateTimeSeries/HeartRateTimeSeries_series2.log logs/HeartRateTimeSeries_HeartRateTimeSeries_series2.log  >  logs/HeartRateTimeSeries_HeartRateTimeSeries_series2.log.diff
	 tail -10  logs/HeartRateTimeSeries_HeartRateTimeSeries_series2.log.diff 

HeartRateTimeSeries/HeartRateTimeSeries_series3.py  :  
	 -$(PYTHON)  tests/HeartRateTimeSeries/HeartRateTimeSeries_series3.py  >  logs/HeartRateTimeSeries_HeartRateTimeSeries_series3.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/HeartRateTimeSeries/HeartRateTimeSeries_series3.log logs/HeartRateTimeSeries_HeartRateTimeSeries_series3.log  >  logs/HeartRateTimeSeries_HeartRateTimeSeries_series3.log.diff
	 tail -10  logs/HeartRateTimeSeries_HeartRateTimeSeries_series3.log.diff 

HeartRateTimeSeries/HeartRateTimeSeries_series4.py  :  
	 -$(PYTHON)  tests/HeartRateTimeSeries/HeartRateTimeSeries_series4.py  >  logs/HeartRateTimeSeries_HeartRateTimeSeries_series4.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/HeartRateTimeSeries/HeartRateTimeSeries_series4.log logs/HeartRateTimeSeries_HeartRateTimeSeries_series4.log  >  logs/HeartRateTimeSeries_HeartRateTimeSeries_series4.log.diff
	 tail -10  logs/HeartRateTimeSeries_HeartRateTimeSeries_series4.log.diff 



 HeartRateTimeSeries :  HeartRateTimeSeries/HeartRateTimeSeries_series4.py HeartRateTimeSeries/HeartRateTimeSeries_series3.py HeartRateTimeSeries/HeartRateTimeSeries_series2.py HeartRateTimeSeries/HeartRateTimeSeries_series1.py  
 

HourOfWeek/test_Business_Hourly_LunchTime.py  :  
	 -$(PYTHON)  tests/HourOfWeek/test_Business_Hourly_LunchTime.py  >  logs/HourOfWeek_test_Business_Hourly_LunchTime.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/HourOfWeek/test_Business_Hourly_LunchTime.log logs/HourOfWeek_test_Business_Hourly_LunchTime.log  >  logs/HourOfWeek_test_Business_Hourly_LunchTime.log.diff
	 tail -10  logs/HourOfWeek_test_Business_Hourly_LunchTime.log.diff 

HourOfWeek/test_Business_Hourly_MondayMorning.py  :  
	 -$(PYTHON)  tests/HourOfWeek/test_Business_Hourly_MondayMorning.py  >  logs/HourOfWeek_test_Business_Hourly_MondayMorning.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/HourOfWeek/test_Business_Hourly_MondayMorning.log logs/HourOfWeek_test_Business_Hourly_MondayMorning.log  >  logs/HourOfWeek_test_Business_Hourly_MondayMorning.log.diff
	 tail -10  logs/HourOfWeek_test_Business_Hourly_MondayMorning.log.diff 

HourOfWeek/test_Business_Hourly_WeekEnd.py  :  
	 -$(PYTHON)  tests/HourOfWeek/test_Business_Hourly_WeekEnd.py  >  logs/HourOfWeek_test_Business_Hourly_WeekEnd.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/HourOfWeek/test_Business_Hourly_WeekEnd.log logs/HourOfWeek_test_Business_Hourly_WeekEnd.log  >  logs/HourOfWeek_test_Business_Hourly_WeekEnd.log.diff
	 tail -10  logs/HourOfWeek_test_Business_Hourly_WeekEnd.log.diff 



 HourOfWeek :  HourOfWeek/test_Business_Hourly_WeekEnd.py HourOfWeek/test_Business_Hourly_MondayMorning.py HourOfWeek/test_Business_Hourly_LunchTime.py  
 

WeekOfMonth/test_Business_DayOfNthWeek.py  :  
	 -$(PYTHON)  tests/WeekOfMonth/test_Business_DayOfNthWeek.py  >  logs/WeekOfMonth_test_Business_DayOfNthWeek.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/WeekOfMonth/test_Business_DayOfNthWeek.log logs/WeekOfMonth_test_Business_DayOfNthWeek.log  >  logs/WeekOfMonth_test_Business_DayOfNthWeek.log.diff
	 tail -10  logs/WeekOfMonth_test_Business_DayOfNthWeek.log.diff 

WeekOfMonth/test_Business_WeekOfMonth.py  :  
	 -$(PYTHON)  tests/WeekOfMonth/test_Business_WeekOfMonth.py  >  logs/WeekOfMonth_test_Business_WeekOfMonth.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/WeekOfMonth/test_Business_WeekOfMonth.log logs/WeekOfMonth_test_Business_WeekOfMonth.log  >  logs/WeekOfMonth_test_Business_WeekOfMonth.log.diff
	 tail -10  logs/WeekOfMonth_test_Business_WeekOfMonth.log.diff 



 WeekOfMonth :  WeekOfMonth/test_Business_WeekOfMonth.py WeekOfMonth/test_Business_DayOfNthWeek.py  
 

artificial/__init__.py  :  
	 -$(PYTHON)  tests/artificial/__init__.py  >  logs/artificial___init__.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/artificial/__init__.log logs/artificial___init__.log  >  logs/artificial___init__.log.diff
	 tail -10  logs/artificial___init__.log.diff 

artificial/process_artificial_dataset.py  :  
	 -$(PYTHON)  tests/artificial/process_artificial_dataset.py  >  logs/artificial_process_artificial_dataset.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/artificial/process_artificial_dataset.log logs/artificial_process_artificial_dataset.log  >  logs/artificial_process_artificial_dataset.log.diff
	 tail -10  logs/artificial_process_artificial_dataset.log.diff 



 artificial :  artificial/process_artificial_dataset.py artificial/__init__.py  
 

basic_checks/hierarchy_checks.py  :  
	 -$(PYTHON)  tests/basic_checks/hierarchy_checks.py  >  logs/basic_checks_hierarchy_checks.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/basic_checks/hierarchy_checks.log logs/basic_checks_hierarchy_checks.log  >  logs/basic_checks_hierarchy_checks.log.diff
	 tail -10  logs/basic_checks_hierarchy_checks.log.diff 

basic_checks/hierarchy_checks_bad_horizon.py  :  
	 -$(PYTHON)  tests/basic_checks/hierarchy_checks_bad_horizon.py  >  logs/basic_checks_hierarchy_checks_bad_horizon.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/basic_checks/hierarchy_checks_bad_horizon.log logs/basic_checks_hierarchy_checks_bad_horizon.log  >  logs/basic_checks_hierarchy_checks_bad_horizon.log.diff
	 tail -10  logs/basic_checks_hierarchy_checks_bad_horizon.log.diff 

basic_checks/hierarchy_checks_base_column_bad_storage.py  :  
	 -$(PYTHON)  tests/basic_checks/hierarchy_checks_base_column_bad_storage.py  >  logs/basic_checks_hierarchy_checks_base_column_bad_storage.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/basic_checks/hierarchy_checks_base_column_bad_storage.log logs/basic_checks_hierarchy_checks_base_column_bad_storage.log  >  logs/basic_checks_hierarchy_checks_base_column_bad_storage.log.diff
	 tail -10  logs/basic_checks_hierarchy_checks_base_column_bad_storage.log.diff 

basic_checks/hierarchy_checks_date_column_bad_storage.py  :  
	 -$(PYTHON)  tests/basic_checks/hierarchy_checks_date_column_bad_storage.py  >  logs/basic_checks_hierarchy_checks_date_column_bad_storage.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/basic_checks/hierarchy_checks_date_column_bad_storage.log logs/basic_checks_hierarchy_checks_date_column_bad_storage.log  >  logs/basic_checks_hierarchy_checks_date_column_bad_storage.log.diff
	 tail -10  logs/basic_checks_hierarchy_checks_date_column_bad_storage.log.diff 

basic_checks/hierarchy_checks_non_existant_base_column.py  :  
	 -$(PYTHON)  tests/basic_checks/hierarchy_checks_non_existant_base_column.py  >  logs/basic_checks_hierarchy_checks_non_existant_base_column.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/basic_checks/hierarchy_checks_non_existant_base_column.log logs/basic_checks_hierarchy_checks_non_existant_base_column.log  >  logs/basic_checks_hierarchy_checks_non_existant_base_column.log.diff
	 tail -10  logs/basic_checks_hierarchy_checks_non_existant_base_column.log.diff 

basic_checks/hierarchy_checks_non_existant_date_column.py  :  
	 -$(PYTHON)  tests/basic_checks/hierarchy_checks_non_existant_date_column.py  >  logs/basic_checks_hierarchy_checks_non_existant_date_column.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/basic_checks/hierarchy_checks_non_existant_date_column.log logs/basic_checks_hierarchy_checks_non_existant_date_column.log  >  logs/basic_checks_hierarchy_checks_non_existant_date_column.log.diff
	 tail -10  logs/basic_checks_hierarchy_checks_non_existant_date_column.log.diff 

basic_checks/hierarchy_checks_unkonwn_base_column.py  :  
	 -$(PYTHON)  tests/basic_checks/hierarchy_checks_unkonwn_base_column.py  >  logs/basic_checks_hierarchy_checks_unkonwn_base_column.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/basic_checks/hierarchy_checks_unkonwn_base_column.log logs/basic_checks_hierarchy_checks_unkonwn_base_column.log  >  logs/basic_checks_hierarchy_checks_unkonwn_base_column.log.diff
	 tail -10  logs/basic_checks_hierarchy_checks_unkonwn_base_column.log.diff 

basic_checks/issue_46_min_max_issues.py  :  
	 -$(PYTHON)  tests/basic_checks/issue_46_min_max_issues.py  >  logs/basic_checks_issue_46_min_max_issues.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/basic_checks/issue_46_min_max_issues.log logs/basic_checks_issue_46_min_max_issues.log  >  logs/basic_checks_issue_46_min_max_issues.log.diff
	 tail -10  logs/basic_checks_issue_46_min_max_issues.log.diff 

basic_checks/issue_46_negative_horizon.py  :  
	 -$(PYTHON)  tests/basic_checks/issue_46_negative_horizon.py  >  logs/basic_checks_issue_46_negative_horizon.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/basic_checks/issue_46_negative_horizon.log logs/basic_checks_issue_46_negative_horizon.log  >  logs/basic_checks_issue_46_negative_horizon.log.diff
	 tail -10  logs/basic_checks_issue_46_negative_horizon.log.diff 

basic_checks/issue_46_non_existant_signal_column.py  :  
	 -$(PYTHON)  tests/basic_checks/issue_46_non_existant_signal_column.py  >  logs/basic_checks_issue_46_non_existant_signal_column.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/basic_checks/issue_46_non_existant_signal_column.log logs/basic_checks_issue_46_non_existant_signal_column.log  >  logs/basic_checks_issue_46_non_existant_signal_column.log.diff
	 tail -10  logs/basic_checks_issue_46_non_existant_signal_column.log.diff 

basic_checks/issue_46_non_existant_time_column.py  :  
	 -$(PYTHON)  tests/basic_checks/issue_46_non_existant_time_column.py  >  logs/basic_checks_issue_46_non_existant_time_column.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/basic_checks/issue_46_non_existant_time_column.log logs/basic_checks_issue_46_non_existant_time_column.log  >  logs/basic_checks_issue_46_non_existant_time_column.log.diff
	 tail -10  logs/basic_checks_issue_46_non_existant_time_column.log.diff 

basic_checks/issue_46_one_or_two_rows.py  :  
	 -$(PYTHON)  tests/basic_checks/issue_46_one_or_two_rows.py  >  logs/basic_checks_issue_46_one_or_two_rows.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/basic_checks/issue_46_one_or_two_rows.log logs/basic_checks_issue_46_one_or_two_rows.log  >  logs/basic_checks_issue_46_one_or_two_rows.log.diff
	 tail -10  logs/basic_checks_issue_46_one_or_two_rows.log.diff 

basic_checks/issue_46_wrong_type_date_column.py  :  
	 -$(PYTHON)  tests/basic_checks/issue_46_wrong_type_date_column.py  >  logs/basic_checks_issue_46_wrong_type_date_column.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/basic_checks/issue_46_wrong_type_date_column.log logs/basic_checks_issue_46_wrong_type_date_column.log  >  logs/basic_checks_issue_46_wrong_type_date_column.log.diff
	 tail -10  logs/basic_checks_issue_46_wrong_type_date_column.log.diff 

basic_checks/issue_46_wrong_type_signal_column.py  :  
	 -$(PYTHON)  tests/basic_checks/issue_46_wrong_type_signal_column.py  >  logs/basic_checks_issue_46_wrong_type_signal_column.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/basic_checks/issue_46_wrong_type_signal_column.log logs/basic_checks_issue_46_wrong_type_signal_column.log  >  logs/basic_checks_issue_46_wrong_type_signal_column.log.diff
	 tail -10  logs/basic_checks_issue_46_wrong_type_signal_column.log.diff 

basic_checks/platform_info.py  :  
	 -$(PYTHON)  tests/basic_checks/platform_info.py  >  logs/basic_checks_platform_info.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/basic_checks/platform_info.log logs/basic_checks_platform_info.log  >  logs/basic_checks_platform_info.log.diff
	 tail -10  logs/basic_checks_platform_info.log.diff 

basic_checks/test_pearson.py  :  
	 -$(PYTHON)  tests/basic_checks/test_pearson.py  >  logs/basic_checks_test_pearson.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/basic_checks/test_pearson.log logs/basic_checks_test_pearson.log  >  logs/basic_checks_test_pearson.log.diff
	 tail -10  logs/basic_checks_test_pearson.log.diff 

basic_checks/test_temporal_demo_hourly_D_H_ErrorMessage.py  :  
	 -$(PYTHON)  tests/basic_checks/test_temporal_demo_hourly_D_H_ErrorMessage.py  >  logs/basic_checks_test_temporal_demo_hourly_D_H_ErrorMessage.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/basic_checks/test_temporal_demo_hourly_D_H_ErrorMessage.log logs/basic_checks_test_temporal_demo_hourly_D_H_ErrorMessage.log  >  logs/basic_checks_test_temporal_demo_hourly_D_H_ErrorMessage.log.diff
	 tail -10  logs/basic_checks_test_temporal_demo_hourly_D_H_ErrorMessage.log.diff 

basic_checks/test_temporal_demo_hourly_D_H_ErrorMessage_2.py  :  
	 -$(PYTHON)  tests/basic_checks/test_temporal_demo_hourly_D_H_ErrorMessage_2.py  >  logs/basic_checks_test_temporal_demo_hourly_D_H_ErrorMessage_2.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/basic_checks/test_temporal_demo_hourly_D_H_ErrorMessage_2.log logs/basic_checks_test_temporal_demo_hourly_D_H_ErrorMessage_2.log  >  logs/basic_checks_test_temporal_demo_hourly_D_H_ErrorMessage_2.log.diff
	 tail -10  logs/basic_checks_test_temporal_demo_hourly_D_H_ErrorMessage_2.log.diff 

basic_checks/test_temporal_demo_hourly_D_H_ErrorMessage_3.py  :  
	 -$(PYTHON)  tests/basic_checks/test_temporal_demo_hourly_D_H_ErrorMessage_3.py  >  logs/basic_checks_test_temporal_demo_hourly_D_H_ErrorMessage_3.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/basic_checks/test_temporal_demo_hourly_D_H_ErrorMessage_3.log logs/basic_checks_test_temporal_demo_hourly_D_H_ErrorMessage_3.log  >  logs/basic_checks_test_temporal_demo_hourly_D_H_ErrorMessage_3.log.diff
	 tail -10  logs/basic_checks_test_temporal_demo_hourly_D_H_ErrorMessage_3.log.diff 



 basic_checks :  basic_checks/test_temporal_demo_hourly_D_H_ErrorMessage_3.py basic_checks/test_temporal_demo_hourly_D_H_ErrorMessage_2.py basic_checks/test_temporal_demo_hourly_D_H_ErrorMessage.py basic_checks/test_pearson.py basic_checks/platform_info.py basic_checks/issue_46_wrong_type_signal_column.py basic_checks/issue_46_wrong_type_date_column.py basic_checks/issue_46_one_or_two_rows.py basic_checks/issue_46_non_existant_time_column.py basic_checks/issue_46_non_existant_signal_column.py basic_checks/issue_46_negative_horizon.py basic_checks/issue_46_min_max_issues.py basic_checks/hierarchy_checks_unkonwn_base_column.py basic_checks/hierarchy_checks_non_existant_date_column.py basic_checks/hierarchy_checks_non_existant_base_column.py basic_checks/hierarchy_checks_date_column_bad_storage.py basic_checks/hierarchy_checks_base_column_bad_storage.py basic_checks/hierarchy_checks_bad_horizon.py basic_checks/hierarchy_checks.py  
 

bugs/run_benchmark_M1Comp_MNB71_18.py  :  
	 -$(PYTHON)  tests/bugs/run_benchmark_M1Comp_MNB71_18.py  >  logs/bugs_run_benchmark_M1Comp_MNB71_18.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/run_benchmark_M1Comp_MNB71_18.log logs/bugs_run_benchmark_M1Comp_MNB71_18.log  >  logs/bugs_run_benchmark_M1Comp_MNB71_18.log.diff
	 tail -10  logs/bugs_run_benchmark_M1Comp_MNB71_18.log.diff 

bugs/run_benchmark_M4Comp_ECONOMICS_ECON1151_18.py  :  
	 -$(PYTHON)  tests/bugs/run_benchmark_M4Comp_ECONOMICS_ECON1151_18.py  >  logs/bugs_run_benchmark_M4Comp_ECONOMICS_ECON1151_18.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/run_benchmark_M4Comp_ECONOMICS_ECON1151_18.log logs/bugs_run_benchmark_M4Comp_ECONOMICS_ECON1151_18.log  >  logs/bugs_run_benchmark_M4Comp_ECONOMICS_ECON1151_18.log.diff
	 tail -10  logs/bugs_run_benchmark_M4Comp_ECONOMICS_ECON1151_18.log.diff 

bugs/run_benchmark_Yahoo_nysecomp_VRS.py  :  
	 -$(PYTHON)  tests/bugs/run_benchmark_Yahoo_nysecomp_VRS.py  >  logs/bugs_run_benchmark_Yahoo_nysecomp_VRS.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/run_benchmark_Yahoo_nysecomp_VRS.log logs/bugs_run_benchmark_Yahoo_nysecomp_VRS.log  >  logs/bugs_run_benchmark_Yahoo_nysecomp_VRS.log.diff
	 tail -10  logs/bugs_run_benchmark_Yahoo_nysecomp_VRS.log.diff 

bugs/test_artificial_bug_1.py  :  
	 -$(PYTHON)  tests/bugs/test_artificial_bug_1.py  >  logs/bugs_test_artificial_bug_1.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/test_artificial_bug_1.log logs/bugs_test_artificial_bug_1.log  >  logs/bugs_test_artificial_bug_1.log.diff
	 tail -10  logs/bugs_test_artificial_bug_1.log.diff 

bugs/test_artificial_bug_2.py  :  
	 -$(PYTHON)  tests/bugs/test_artificial_bug_2.py  >  logs/bugs_test_artificial_bug_2.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/test_artificial_bug_2.log logs/bugs_test_artificial_bug_2.log  >  logs/bugs_test_artificial_bug_2.log.diff
	 tail -10  logs/bugs_test_artificial_bug_2.log.diff 

bugs/test_random_exogenous.py  :  
	 -$(PYTHON)  tests/bugs/test_random_exogenous.py  >  logs/bugs_test_random_exogenous.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/test_random_exogenous.log logs/bugs_test_random_exogenous.log  >  logs/bugs_test_random_exogenous.log.diff
	 tail -10  logs/bugs_test_random_exogenous.log.diff 

bugs/issue_106/insurance_exog.py  :  
	 -$(PYTHON)  tests/bugs/issue_106/insurance_exog.py  >  logs/bugs_issue_106_insurance_exog.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_106/insurance_exog.log logs/bugs_issue_106_insurance_exog.log  >  logs/bugs_issue_106_insurance_exog.log.diff
	 tail -10  logs/bugs_issue_106_insurance_exog.log.diff 

bugs/issue_106/insurance_exog_svrx.py  :  
	 -$(PYTHON)  tests/bugs/issue_106/insurance_exog_svrx.py  >  logs/bugs_issue_106_insurance_exog_svrx.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_106/insurance_exog_svrx.log logs/bugs_issue_106_insurance_exog_svrx.log  >  logs/bugs_issue_106_insurance_exog_svrx.log.diff
	 tail -10  logs/bugs_issue_106_insurance_exog_svrx.log.diff 

bugs/issue_106/insurance_exog_xgbx.py  :  
	 -$(PYTHON)  tests/bugs/issue_106/insurance_exog_xgbx.py  >  logs/bugs_issue_106_insurance_exog_xgbx.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_106/insurance_exog_xgbx.log logs/bugs_issue_106_insurance_exog_xgbx.log  >  logs/bugs_issue_106_insurance_exog_xgbx.log.diff
	 tail -10  logs/bugs_issue_106_insurance_exog_xgbx.log.diff 

bugs/issue_106/submitted_script.py  :  
	 -$(PYTHON)  tests/bugs/issue_106/submitted_script.py  >  logs/bugs_issue_106_submitted_script.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_106/submitted_script.log logs/bugs_issue_106_submitted_script.log  >  logs/bugs_issue_106_submitted_script.log.diff
	 tail -10  logs/bugs_issue_106_submitted_script.log.diff 

bugs/issue_179/issue_179_cumprod_overflow.py  :  
	 -$(PYTHON)  tests/bugs/issue_179/issue_179_cumprod_overflow.py  >  logs/bugs_issue_179_issue_179_cumprod_overflow.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_179/issue_179_cumprod_overflow.log logs/bugs_issue_179_issue_179_cumprod_overflow.log  >  logs/bugs_issue_179_issue_179_cumprod_overflow.log.diff
	 tail -10  logs/bugs_issue_179_issue_179_cumprod_overflow.log.diff 

bugs/issue_19/issue_19_no_interpolation.py  :  
	 -$(PYTHON)  tests/bugs/issue_19/issue_19_no_interpolation.py  >  logs/bugs_issue_19_issue_19_no_interpolation.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_19/issue_19_no_interpolation.log logs/bugs_issue_19_issue_19_no_interpolation.log  >  logs/bugs_issue_19_issue_19_no_interpolation.log.diff
	 tail -10  logs/bugs_issue_19_issue_19_no_interpolation.log.diff 

bugs/issue_193/test_ozone_exogenous_issue_193_no_exog.py  :  
	 -$(PYTHON)  tests/bugs/issue_193/test_ozone_exogenous_issue_193_no_exog.py  >  logs/bugs_issue_193_test_ozone_exogenous_issue_193_no_exog.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_193/test_ozone_exogenous_issue_193_no_exog.log logs/bugs_issue_193_test_ozone_exogenous_issue_193_no_exog.log  >  logs/bugs_issue_193_test_ozone_exogenous_issue_193_no_exog.log.diff
	 tail -10  logs/bugs_issue_193_test_ozone_exogenous_issue_193_no_exog.log.diff 

bugs/issue_193/test_ozone_exogenous_issue_193_with_exog_arx_disabled.py  :  
	 -$(PYTHON)  tests/bugs/issue_193/test_ozone_exogenous_issue_193_with_exog_arx_disabled.py  >  logs/bugs_issue_193_test_ozone_exogenous_issue_193_with_exog_arx_disabled.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_193/test_ozone_exogenous_issue_193_with_exog_arx_disabled.log logs/bugs_issue_193_test_ozone_exogenous_issue_193_with_exog_arx_disabled.log  >  logs/bugs_issue_193_test_ozone_exogenous_issue_193_with_exog_arx_disabled.log.diff
	 tail -10  logs/bugs_issue_193_test_ozone_exogenous_issue_193_with_exog_arx_disabled.log.diff 

bugs/issue_193/test_ozone_exogenous_issue_193_with_exog_arx_enabled.py  :  
	 -$(PYTHON)  tests/bugs/issue_193/test_ozone_exogenous_issue_193_with_exog_arx_enabled.py  >  logs/bugs_issue_193_test_ozone_exogenous_issue_193_with_exog_arx_enabled.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_193/test_ozone_exogenous_issue_193_with_exog_arx_enabled.log logs/bugs_issue_193_test_ozone_exogenous_issue_193_with_exog_arx_enabled.log  >  logs/bugs_issue_193_test_ozone_exogenous_issue_193_with_exog_arx_enabled.log.diff
	 tail -10  logs/bugs_issue_193_test_ozone_exogenous_issue_193_with_exog_arx_enabled.log.diff 

bugs/issue_21/test_artificial_1024__poly_7__20.py  :  
	 -$(PYTHON)  tests/bugs/issue_21/test_artificial_1024__poly_7__20.py  >  logs/bugs_issue_21_test_artificial_1024__poly_7__20.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_21/test_artificial_1024__poly_7__20.log logs/bugs_issue_21_test_artificial_1024__poly_7__20.log  >  logs/bugs_issue_21_test_artificial_1024__poly_7__20.log.diff
	 tail -10  logs/bugs_issue_21_test_artificial_1024__poly_7__20.log.diff 

bugs/issue_21/test_ozone_max_autoreg_5.py  :  
	 -$(PYTHON)  tests/bugs/issue_21/test_ozone_max_autoreg_5.py  >  logs/bugs_issue_21_test_ozone_max_autoreg_5.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_21/test_ozone_max_autoreg_5.log logs/bugs_issue_21_test_ozone_max_autoreg_5.log  >  logs/bugs_issue_21_test_ozone_max_autoreg_5.log.diff
	 tail -10  logs/bugs_issue_21_test_ozone_max_autoreg_5.log.diff 

bugs/issue_216/issue_216_prediction_interval_shaded_area.py  :  
	 -$(PYTHON)  tests/bugs/issue_216/issue_216_prediction_interval_shaded_area.py  >  logs/bugs_issue_216_issue_216_prediction_interval_shaded_area.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_216/issue_216_prediction_interval_shaded_area.log logs/bugs_issue_216_issue_216_prediction_interval_shaded_area.log  >  logs/bugs_issue_216_issue_216_prediction_interval_shaded_area.log.diff
	 tail -10  logs/bugs_issue_216_issue_216_prediction_interval_shaded_area.log.diff 

bugs/issue_22/test_artificial_1024_cumsum_constant_5__20.py  :  
	 -$(PYTHON)  tests/bugs/issue_22/test_artificial_1024_cumsum_constant_5__20.py  >  logs/bugs_issue_22_test_artificial_1024_cumsum_constant_5__20.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_22/test_artificial_1024_cumsum_constant_5__20.log logs/bugs_issue_22_test_artificial_1024_cumsum_constant_5__20.log  >  logs/bugs_issue_22_test_artificial_1024_cumsum_constant_5__20.log.diff
	 tail -10  logs/bugs_issue_22_test_artificial_1024_cumsum_constant_5__20.log.diff 

bugs/issue_220/test_ozone_Lag1Trend_bug.py  :  
	 -$(PYTHON)  tests/bugs/issue_220/test_ozone_Lag1Trend_bug.py  >  logs/bugs_issue_220_test_ozone_Lag1Trend_bug.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_220/test_ozone_Lag1Trend_bug.log logs/bugs_issue_220_test_ozone_Lag1Trend_bug.log  >  logs/bugs_issue_220_test_ozone_Lag1Trend_bug.log.diff
	 tail -10  logs/bugs_issue_220_test_ozone_Lag1Trend_bug.log.diff 

bugs/issue_29/test_mem_1.py  :  
	 -$(PYTHON)  tests/bugs/issue_29/test_mem_1.py  >  logs/bugs_issue_29_test_mem_1.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_29/test_mem_1.log logs/bugs_issue_29_test_mem_1.log  >  logs/bugs_issue_29_test_mem_1.log.diff
	 tail -10  logs/bugs_issue_29_test_mem_1.log.diff 

bugs/issue_32/bug_test_ozone_heroku_1.py  :  
	 -$(PYTHON)  tests/bugs/issue_32/bug_test_ozone_heroku_1.py  >  logs/bugs_issue_32_bug_test_ozone_heroku_1.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_32/bug_test_ozone_heroku_1.log logs/bugs_issue_32_bug_test_ozone_heroku_1.log  >  logs/bugs_issue_32_bug_test_ozone_heroku_1.log.diff
	 tail -10  logs/bugs_issue_32_bug_test_ozone_heroku_1.log.diff 

bugs/issue_32/bug_test_ozone_heroku_1_ws.py  :  
	 -$(PYTHON)  tests/bugs/issue_32/bug_test_ozone_heroku_1_ws.py  >  logs/bugs_issue_32_bug_test_ozone_heroku_1_ws.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_32/bug_test_ozone_heroku_1_ws.log logs/bugs_issue_32_bug_test_ozone_heroku_1_ws.log  >  logs/bugs_issue_32_bug_test_ozone_heroku_1_ws.log.diff
	 tail -10  logs/bugs_issue_32_bug_test_ozone_heroku_1_ws.log.diff 

bugs/issue_34/issue_34.py  :  
	 -$(PYTHON)  tests/bugs/issue_34/issue_34.py  >  logs/bugs_issue_34_issue_34.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_34/issue_34.log logs/bugs_issue_34_issue_34.log  >  logs/bugs_issue_34_issue_34.log.diff
	 tail -10  logs/bugs_issue_34_issue_34.log.diff 

bugs/issue_34/issue_34_1.py  :  
	 -$(PYTHON)  tests/bugs/issue_34/issue_34_1.py  >  logs/bugs_issue_34_issue_34_1.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_34/issue_34_1.log logs/bugs_issue_34_issue_34_1.log  >  logs/bugs_issue_34_issue_34_1.log.diff
	 tail -10  logs/bugs_issue_34_issue_34_1.log.diff 

bugs/issue_34/test_artificial_1024__poly_7_12_100.py  :  
	 -$(PYTHON)  tests/bugs/issue_34/test_artificial_1024__poly_7_12_100.py  >  logs/bugs_issue_34_test_artificial_1024__poly_7_12_100.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_34/test_artificial_1024__poly_7_12_100.log logs/bugs_issue_34_test_artificial_1024__poly_7_12_100.log  >  logs/bugs_issue_34_test_artificial_1024__poly_7_12_100.log.diff
	 tail -10  logs/bugs_issue_34_test_artificial_1024__poly_7_12_100.log.diff 

bugs/issue_34/test_artificial_1024_diff_poly_0__0.py  :  
	 -$(PYTHON)  tests/bugs/issue_34/test_artificial_1024_diff_poly_0__0.py  >  logs/bugs_issue_34_test_artificial_1024_diff_poly_0__0.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_34/test_artificial_1024_diff_poly_0__0.log logs/bugs_issue_34_test_artificial_1024_diff_poly_0__0.log  >  logs/bugs_issue_34_test_artificial_1024_diff_poly_0__0.log.diff
	 tail -10  logs/bugs_issue_34_test_artificial_1024_diff_poly_0__0.log.diff 

bugs/issue_34/test_artificial_1024_diff_poly_0__100.py  :  
	 -$(PYTHON)  tests/bugs/issue_34/test_artificial_1024_diff_poly_0__100.py  >  logs/bugs_issue_34_test_artificial_1024_diff_poly_0__100.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_34/test_artificial_1024_diff_poly_0__100.log logs/bugs_issue_34_test_artificial_1024_diff_poly_0__100.log  >  logs/bugs_issue_34_test_artificial_1024_diff_poly_0__100.log.diff
	 tail -10  logs/bugs_issue_34_test_artificial_1024_diff_poly_0__100.log.diff 

bugs/issue_34/test_artificial_1024_inv_constant_30__20.py  :  
	 -$(PYTHON)  tests/bugs/issue_34/test_artificial_1024_inv_constant_30__20.py  >  logs/bugs_issue_34_test_artificial_1024_inv_constant_30__20.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_34/test_artificial_1024_inv_constant_30__20.log logs/bugs_issue_34_test_artificial_1024_inv_constant_30__20.log  >  logs/bugs_issue_34_test_artificial_1024_inv_constant_30__20.log.diff
	 tail -10  logs/bugs_issue_34_test_artificial_1024_inv_constant_30__20.log.diff 

bugs/issue_34/test_artificial_1024_inv_linear_7_12_100.py  :  
	 -$(PYTHON)  tests/bugs/issue_34/test_artificial_1024_inv_linear_7_12_100.py  >  logs/bugs_issue_34_test_artificial_1024_inv_linear_7_12_100.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_34/test_artificial_1024_inv_linear_7_12_100.log logs/bugs_issue_34_test_artificial_1024_inv_linear_7_12_100.log  >  logs/bugs_issue_34_test_artificial_1024_inv_linear_7_12_100.log.diff
	 tail -10  logs/bugs_issue_34_test_artificial_1024_inv_linear_7_12_100.log.diff 

bugs/issue_34/test_artificial_1024_log_linear_30_12_20.py  :  
	 -$(PYTHON)  tests/bugs/issue_34/test_artificial_1024_log_linear_30_12_20.py  >  logs/bugs_issue_34_test_artificial_1024_log_linear_30_12_20.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_34/test_artificial_1024_log_linear_30_12_20.log logs/bugs_issue_34_test_artificial_1024_log_linear_30_12_20.log  >  logs/bugs_issue_34_test_artificial_1024_log_linear_30_12_20.log.diff
	 tail -10  logs/bugs_issue_34_test_artificial_1024_log_linear_30_12_20.log.diff 

bugs/issue_34/test_artificial_1024_log_linear_5_12_100.py  :  
	 -$(PYTHON)  tests/bugs/issue_34/test_artificial_1024_log_linear_5_12_100.py  >  logs/bugs_issue_34_test_artificial_1024_log_linear_5_12_100.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_34/test_artificial_1024_log_linear_5_12_100.log logs/bugs_issue_34_test_artificial_1024_log_linear_5_12_100.log  >  logs/bugs_issue_34_test_artificial_1024_log_linear_5_12_100.log.diff
	 tail -10  logs/bugs_issue_34_test_artificial_1024_log_linear_5_12_100.log.diff 

bugs/issue_34/test_artificial_1024_sqr_poly_5_12_20.py  :  
	 -$(PYTHON)  tests/bugs/issue_34/test_artificial_1024_sqr_poly_5_12_20.py  >  logs/bugs_issue_34_test_artificial_1024_sqr_poly_5_12_20.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_34/test_artificial_1024_sqr_poly_5_12_20.log logs/bugs_issue_34_test_artificial_1024_sqr_poly_5_12_20.log  >  logs/bugs_issue_34_test_artificial_1024_sqr_poly_5_12_20.log.diff
	 tail -10  logs/bugs_issue_34_test_artificial_1024_sqr_poly_5_12_20.log.diff 

bugs/issue_34/test_artificial_1024_sqrt_constant_7_12_100.py  :  
	 -$(PYTHON)  tests/bugs/issue_34/test_artificial_1024_sqrt_constant_7_12_100.py  >  logs/bugs_issue_34_test_artificial_1024_sqrt_constant_7_12_100.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_34/test_artificial_1024_sqrt_constant_7_12_100.log logs/bugs_issue_34_test_artificial_1024_sqrt_constant_7_12_100.log  >  logs/bugs_issue_34_test_artificial_1024_sqrt_constant_7_12_100.log.diff
	 tail -10  logs/bugs_issue_34_test_artificial_1024_sqrt_constant_7_12_100.log.diff 

bugs/issue_34/test_artificial_1024_sqrt_linear_30_12_20.py  :  
	 -$(PYTHON)  tests/bugs/issue_34/test_artificial_1024_sqrt_linear_30_12_20.py  >  logs/bugs_issue_34_test_artificial_1024_sqrt_linear_30_12_20.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_34/test_artificial_1024_sqrt_linear_30_12_20.log logs/bugs_issue_34_test_artificial_1024_sqrt_linear_30_12_20.log  >  logs/bugs_issue_34_test_artificial_1024_sqrt_linear_30_12_20.log.diff
	 tail -10  logs/bugs_issue_34_test_artificial_1024_sqrt_linear_30_12_20.log.diff 

bugs/issue_36/display_version_info.py  :  
	 -$(PYTHON)  tests/bugs/issue_36/display_version_info.py  >  logs/bugs_issue_36_display_version_info.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_36/display_version_info.log logs/bugs_issue_36_display_version_info.log  >  logs/bugs_issue_36_display_version_info.log.diff
	 tail -10  logs/bugs_issue_36_display_version_info.log.diff 

bugs/issue_36/issue_36_version_info.py  :  
	 -$(PYTHON)  tests/bugs/issue_36/issue_36_version_info.py  >  logs/bugs_issue_36_issue_36_version_info.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_36/issue_36_version_info.log logs/bugs_issue_36_issue_36_version_info.log  >  logs/bugs_issue_36_issue_36_version_info.log.diff
	 tail -10  logs/bugs_issue_36_issue_36_version_info.log.diff 

bugs/issue_4/issue_4.py  :  
	 -$(PYTHON)  tests/bugs/issue_4/issue_4.py  >  logs/bugs_issue_4_issue_4.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_4/issue_4.log logs/bugs_issue_4_issue_4.log  >  logs/bugs_issue_4_issue_4.log.diff
	 tail -10  logs/bugs_issue_4_issue_4.log.diff 

bugs/issue_46/issue_46_one_or_two_rows.py  :  
	 -$(PYTHON)  tests/bugs/issue_46/issue_46_one_or_two_rows.py  >  logs/bugs_issue_46_issue_46_one_or_two_rows.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_46/issue_46_one_or_two_rows.log logs/bugs_issue_46_issue_46_one_or_two_rows.log  >  logs/bugs_issue_46_issue_46_one_or_two_rows.log.diff
	 tail -10  logs/bugs_issue_46_issue_46_one_or_two_rows.log.diff 

bugs/issue_55/grouping_issue_55_notebook.py  :  
	 -$(PYTHON)  tests/bugs/issue_55/grouping_issue_55_notebook.py  >  logs/bugs_issue_55_grouping_issue_55_notebook.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_55/grouping_issue_55_notebook.log logs/bugs_issue_55_grouping_issue_55_notebook.log  >  logs/bugs_issue_55_grouping_issue_55_notebook.log.diff
	 tail -10  logs/bugs_issue_55_grouping_issue_55_notebook.log.diff 

bugs/issue_56/issue_56_order1.py  :  
	 -$(PYTHON)  tests/bugs/issue_56/issue_56_order1.py  >  logs/bugs_issue_56_issue_56_order1.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_56/issue_56_order1.log logs/bugs_issue_56_issue_56_order1.log  >  logs/bugs_issue_56_issue_56_order1.log.diff
	 tail -10  logs/bugs_issue_56_issue_56_order1.log.diff 

bugs/issue_56/issue_56_order2.py  :  
	 -$(PYTHON)  tests/bugs/issue_56/issue_56_order2.py  >  logs/bugs_issue_56_issue_56_order2.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_56/issue_56_order2.log logs/bugs_issue_56_issue_56_order2.log  >  logs/bugs_issue_56_issue_56_order2.log.diff
	 tail -10  logs/bugs_issue_56_issue_56_order2.log.diff 

bugs/issue_58/issue_58_1_categorical_exogenous.py  :  
	 -$(PYTHON)  tests/bugs/issue_58/issue_58_1_categorical_exogenous.py  >  logs/bugs_issue_58_issue_58_1_categorical_exogenous.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_58/issue_58_1_categorical_exogenous.log logs/bugs_issue_58_issue_58_1_categorical_exogenous.log  >  logs/bugs_issue_58_issue_58_1_categorical_exogenous.log.diff
	 tail -10  logs/bugs_issue_58_issue_58_1_categorical_exogenous.log.diff 

bugs/issue_69/issue_69_example_1_no_rectifier.py  :  
	 -$(PYTHON)  tests/bugs/issue_69/issue_69_example_1_no_rectifier.py  >  logs/bugs_issue_69_issue_69_example_1_no_rectifier.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_69/issue_69_example_1_no_rectifier.log logs/bugs_issue_69_issue_69_example_1_no_rectifier.log  >  logs/bugs_issue_69_issue_69_example_1_no_rectifier.log.diff
	 tail -10  logs/bugs_issue_69_issue_69_example_1_no_rectifier.log.diff 

bugs/issue_69/issue_69_example_1_with_rectifier.py  :  
	 -$(PYTHON)  tests/bugs/issue_69/issue_69_example_1_with_rectifier.py  >  logs/bugs_issue_69_issue_69_example_1_with_rectifier.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_69/issue_69_example_1_with_rectifier.log logs/bugs_issue_69_issue_69_example_1_with_rectifier.log  >  logs/bugs_issue_69_issue_69_example_1_with_rectifier.log.diff
	 tail -10  logs/bugs_issue_69_issue_69_example_1_with_rectifier.log.diff 

bugs/issue_70/test_artificial_filter_seasonals_day.py  :  
	 -$(PYTHON)  tests/bugs/issue_70/test_artificial_filter_seasonals_day.py  >  logs/bugs_issue_70_test_artificial_filter_seasonals_day.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_70/test_artificial_filter_seasonals_day.log logs/bugs_issue_70_test_artificial_filter_seasonals_day.log  >  logs/bugs_issue_70_test_artificial_filter_seasonals_day.log.diff
	 tail -10  logs/bugs_issue_70_test_artificial_filter_seasonals_day.log.diff 

bugs/issue_70/test_artificial_filter_seasonals_hour.py  :  
	 -$(PYTHON)  tests/bugs/issue_70/test_artificial_filter_seasonals_hour.py  >  logs/bugs_issue_70_test_artificial_filter_seasonals_hour.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_70/test_artificial_filter_seasonals_hour.log logs/bugs_issue_70_test_artificial_filter_seasonals_hour.log  >  logs/bugs_issue_70_test_artificial_filter_seasonals_hour.log.diff
	 tail -10  logs/bugs_issue_70_test_artificial_filter_seasonals_hour.log.diff 

bugs/issue_70/test_artificial_filter_seasonals_min.py  :  
	 -$(PYTHON)  tests/bugs/issue_70/test_artificial_filter_seasonals_min.py  >  logs/bugs_issue_70_test_artificial_filter_seasonals_min.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_70/test_artificial_filter_seasonals_min.log logs/bugs_issue_70_test_artificial_filter_seasonals_min.log  >  logs/bugs_issue_70_test_artificial_filter_seasonals_min.log.diff
	 tail -10  logs/bugs_issue_70_test_artificial_filter_seasonals_min.log.diff 

bugs/issue_70/test_artificial_filter_seasonals_second.py  :  
	 -$(PYTHON)  tests/bugs/issue_70/test_artificial_filter_seasonals_second.py  >  logs/bugs_issue_70_test_artificial_filter_seasonals_second.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_70/test_artificial_filter_seasonals_second.log logs/bugs_issue_70_test_artificial_filter_seasonals_second.log  >  logs/bugs_issue_70_test_artificial_filter_seasonals_second.log.diff
	 tail -10  logs/bugs_issue_70_test_artificial_filter_seasonals_second.log.diff 

bugs/issue_70/test_artificial_keep_all_seasonals_day.py  :  
	 -$(PYTHON)  tests/bugs/issue_70/test_artificial_keep_all_seasonals_day.py  >  logs/bugs_issue_70_test_artificial_keep_all_seasonals_day.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_70/test_artificial_keep_all_seasonals_day.log logs/bugs_issue_70_test_artificial_keep_all_seasonals_day.log  >  logs/bugs_issue_70_test_artificial_keep_all_seasonals_day.log.diff
	 tail -10  logs/bugs_issue_70_test_artificial_keep_all_seasonals_day.log.diff 

bugs/issue_70/test_artificial_keep_all_seasonals_hour.py  :  
	 -$(PYTHON)  tests/bugs/issue_70/test_artificial_keep_all_seasonals_hour.py  >  logs/bugs_issue_70_test_artificial_keep_all_seasonals_hour.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_70/test_artificial_keep_all_seasonals_hour.log logs/bugs_issue_70_test_artificial_keep_all_seasonals_hour.log  >  logs/bugs_issue_70_test_artificial_keep_all_seasonals_hour.log.diff
	 tail -10  logs/bugs_issue_70_test_artificial_keep_all_seasonals_hour.log.diff 

bugs/issue_70/test_artificial_keep_all_seasonals_minute.py  :  
	 -$(PYTHON)  tests/bugs/issue_70/test_artificial_keep_all_seasonals_minute.py  >  logs/bugs_issue_70_test_artificial_keep_all_seasonals_minute.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_70/test_artificial_keep_all_seasonals_minute.log logs/bugs_issue_70_test_artificial_keep_all_seasonals_minute.log  >  logs/bugs_issue_70_test_artificial_keep_all_seasonals_minute.log.diff
	 tail -10  logs/bugs_issue_70_test_artificial_keep_all_seasonals_minute.log.diff 

bugs/issue_70/test_artificial_keep_all_seasonals_second.py  :  
	 -$(PYTHON)  tests/bugs/issue_70/test_artificial_keep_all_seasonals_second.py  >  logs/bugs_issue_70_test_artificial_keep_all_seasonals_second.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_70/test_artificial_keep_all_seasonals_second.log logs/bugs_issue_70_test_artificial_keep_all_seasonals_second.log  >  logs/bugs_issue_70_test_artificial_keep_all_seasonals_second.log.diff
	 tail -10  logs/bugs_issue_70_test_artificial_keep_all_seasonals_second.log.diff 

bugs/issue_70/test_ozone_filter_seasonals.py  :  
	 -$(PYTHON)  tests/bugs/issue_70/test_ozone_filter_seasonals.py  >  logs/bugs_issue_70_test_ozone_filter_seasonals.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_70/test_ozone_filter_seasonals.log logs/bugs_issue_70_test_ozone_filter_seasonals.log  >  logs/bugs_issue_70_test_ozone_filter_seasonals.log.diff
	 tail -10  logs/bugs_issue_70_test_ozone_filter_seasonals.log.diff 

bugs/issue_70/test_ozone_keep_all_seasonals.py  :  
	 -$(PYTHON)  tests/bugs/issue_70/test_ozone_keep_all_seasonals.py  >  logs/bugs_issue_70_test_ozone_keep_all_seasonals.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_70/test_ozone_keep_all_seasonals.log logs/bugs_issue_70_test_ozone_keep_all_seasonals.log  >  logs/bugs_issue_70_test_ozone_keep_all_seasonals.log.diff
	 tail -10  logs/bugs_issue_70_test_ozone_keep_all_seasonals.log.diff 

bugs/issue_72/test_artificial_32_exp_linear_0__100.py  :  
	 -$(PYTHON)  tests/bugs/issue_72/test_artificial_32_exp_linear_0__100.py  >  logs/bugs_issue_72_test_artificial_32_exp_linear_0__100.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_72/test_artificial_32_exp_linear_0__100.log logs/bugs_issue_72_test_artificial_32_exp_linear_0__100.log  >  logs/bugs_issue_72_test_artificial_32_exp_linear_0__100.log.diff
	 tail -10  logs/bugs_issue_72_test_artificial_32_exp_linear_0__100.log.diff 

bugs/issue_72/test_artificial_32_sqrt_poly_7__20.py  :  
	 -$(PYTHON)  tests/bugs/issue_72/test_artificial_32_sqrt_poly_7__20.py  >  logs/bugs_issue_72_test_artificial_32_sqrt_poly_7__20.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_72/test_artificial_32_sqrt_poly_7__20.log logs/bugs_issue_72_test_artificial_32_sqrt_poly_7__20.log  >  logs/bugs_issue_72_test_artificial_32_sqrt_poly_7__20.log.diff
	 tail -10  logs/bugs_issue_72_test_artificial_32_sqrt_poly_7__20.log.diff 

bugs/issue_73/issue_73_1_fast_mode.py  :  
	 -$(PYTHON)  tests/bugs/issue_73/issue_73_1_fast_mode.py  >  logs/bugs_issue_73_issue_73_1_fast_mode.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_73/issue_73_1_fast_mode.log logs/bugs_issue_73_issue_73_1_fast_mode.log  >  logs/bugs_issue_73_issue_73_1_fast_mode.log.diff
	 tail -10  logs/bugs_issue_73_issue_73_1_fast_mode.log.diff 

bugs/issue_73/issue_73_1_fast_mode_2.py  :  
	 -$(PYTHON)  tests/bugs/issue_73/issue_73_1_fast_mode_2.py  >  logs/bugs_issue_73_issue_73_1_fast_mode_2.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_73/issue_73_1_fast_mode_2.log logs/bugs_issue_73_issue_73_1_fast_mode_2.log  >  logs/bugs_issue_73_issue_73_1_fast_mode_2.log.diff
	 tail -10  logs/bugs_issue_73_issue_73_1_fast_mode_2.log.diff 

bugs/issue_75/issue_75_plot.py  :  
	 -$(PYTHON)  tests/bugs/issue_75/issue_75_plot.py  >  logs/bugs_issue_75_issue_75_plot.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_75/issue_75_plot.log logs/bugs_issue_75_issue_75_plot.log  >  logs/bugs_issue_75_issue_75_plot.log.diff
	 tail -10  logs/bugs_issue_75_issue_75_plot.log.diff 

bugs/issue_75/issue_75_residues_Are_empty.py  :  
	 -$(PYTHON)  tests/bugs/issue_75/issue_75_residues_Are_empty.py  >  logs/bugs_issue_75_issue_75_residues_Are_empty.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_75/issue_75_residues_Are_empty.log logs/bugs_issue_75_issue_75_residues_Are_empty.log  >  logs/bugs_issue_75_issue_75_residues_Are_empty.log.diff
	 tail -10  logs/bugs_issue_75_issue_75_residues_Are_empty.log.diff 

bugs/issue_76/test_ozone_unicode.py  :  
	 -$(PYTHON)  tests/bugs/issue_76/test_ozone_unicode.py  >  logs/bugs_issue_76_test_ozone_unicode.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_76/test_ozone_unicode.log logs/bugs_issue_76_test_ozone_unicode.log  >  logs/bugs_issue_76_test_ozone_unicode.log.diff
	 tail -10  logs/bugs_issue_76_test_ozone_unicode.log.diff 

bugs/issue_79/test_ozone_missing_signal.py  :  
	 -$(PYTHON)  tests/bugs/issue_79/test_ozone_missing_signal.py  >  logs/bugs_issue_79_test_ozone_missing_signal.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_79/test_ozone_missing_signal.log logs/bugs_issue_79_test_ozone_missing_signal.log  >  logs/bugs_issue_79_test_ozone_missing_signal.log.diff
	 tail -10  logs/bugs_issue_79_test_ozone_missing_signal.log.diff 

bugs/issue_80/test_ozone_missing_time.py  :  
	 -$(PYTHON)  tests/bugs/issue_80/test_ozone_missing_time.py  >  logs/bugs_issue_80_test_ozone_missing_time.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_80/test_ozone_missing_time.log logs/bugs_issue_80_test_ozone_missing_time.log  >  logs/bugs_issue_80_test_ozone_missing_time.log.diff
	 tail -10  logs/bugs_issue_80_test_ozone_missing_time.log.diff 

bugs/issue_82/issue_82_long_cycles.py  :  
	 -$(PYTHON)  tests/bugs/issue_82/issue_82_long_cycles.py  >  logs/bugs_issue_82_issue_82_long_cycles.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_82/issue_82_long_cycles.log logs/bugs_issue_82_issue_82_long_cycles.log  >  logs/bugs_issue_82_issue_82_long_cycles.log.diff
	 tail -10  logs/bugs_issue_82_issue_82_long_cycles.log.diff 

bugs/issue_94/issue_94.py  :  
	 -$(PYTHON)  tests/bugs/issue_94/issue_94.py  >  logs/bugs_issue_94_issue_94.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/bugs/issue_94/issue_94.log logs/bugs_issue_94_issue_94.log  >  logs/bugs_issue_94_issue_94.log.diff
	 tail -10  logs/bugs_issue_94_issue_94.log.diff 



 bugs :  bugs/test_random_exogenous.py bugs/test_artificial_bug_2.py bugs/test_artificial_bug_1.py bugs/run_benchmark_Yahoo_nysecomp_VRS.py bugs/run_benchmark_M4Comp_ECONOMICS_ECON1151_18.py bugs/run_benchmark_M1Comp_MNB71_18.py bugs/issue_106/submitted_script.py bugs/issue_106/insurance_exog_xgbx.py bugs/issue_106/insurance_exog_svrx.py bugs/issue_106/insurance_exog.py bugs/issue_179/issue_179_cumprod_overflow.py bugs/issue_19/issue_19_no_interpolation.py bugs/issue_193/test_ozone_exogenous_issue_193_with_exog_arx_enabled.py bugs/issue_193/test_ozone_exogenous_issue_193_with_exog_arx_disabled.py bugs/issue_193/test_ozone_exogenous_issue_193_no_exog.py bugs/issue_21/test_ozone_max_autoreg_5.py bugs/issue_21/test_artificial_1024__poly_7__20.py bugs/issue_216/issue_216_prediction_interval_shaded_area.py bugs/issue_22/test_artificial_1024_cumsum_constant_5__20.py bugs/issue_220/test_ozone_Lag1Trend_bug.py bugs/issue_29/test_mem_1.py bugs/issue_32/bug_test_ozone_heroku_1_ws.py bugs/issue_32/bug_test_ozone_heroku_1.py bugs/issue_34/test_artificial_1024_sqrt_linear_30_12_20.py bugs/issue_34/test_artificial_1024_sqrt_constant_7_12_100.py bugs/issue_34/test_artificial_1024_sqr_poly_5_12_20.py bugs/issue_34/test_artificial_1024_log_linear_5_12_100.py bugs/issue_34/test_artificial_1024_log_linear_30_12_20.py bugs/issue_34/test_artificial_1024_inv_linear_7_12_100.py bugs/issue_34/test_artificial_1024_inv_constant_30__20.py bugs/issue_34/test_artificial_1024_diff_poly_0__100.py bugs/issue_34/test_artificial_1024_diff_poly_0__0.py bugs/issue_34/test_artificial_1024__poly_7_12_100.py bugs/issue_34/issue_34_1.py bugs/issue_34/issue_34.py bugs/issue_36/issue_36_version_info.py bugs/issue_36/display_version_info.py bugs/issue_4/issue_4.py bugs/issue_46/issue_46_one_or_two_rows.py bugs/issue_55/grouping_issue_55_notebook.py bugs/issue_56/issue_56_order2.py bugs/issue_56/issue_56_order1.py bugs/issue_58/issue_58_1_categorical_exogenous.py bugs/issue_69/issue_69_example_1_with_rectifier.py bugs/issue_69/issue_69_example_1_no_rectifier.py bugs/issue_70/test_ozone_keep_all_seasonals.py bugs/issue_70/test_ozone_filter_seasonals.py bugs/issue_70/test_artificial_keep_all_seasonals_second.py bugs/issue_70/test_artificial_keep_all_seasonals_minute.py bugs/issue_70/test_artificial_keep_all_seasonals_hour.py bugs/issue_70/test_artificial_keep_all_seasonals_day.py bugs/issue_70/test_artificial_filter_seasonals_second.py bugs/issue_70/test_artificial_filter_seasonals_min.py bugs/issue_70/test_artificial_filter_seasonals_hour.py bugs/issue_70/test_artificial_filter_seasonals_day.py bugs/issue_72/test_artificial_32_sqrt_poly_7__20.py bugs/issue_72/test_artificial_32_exp_linear_0__100.py bugs/issue_73/issue_73_1_fast_mode_2.py bugs/issue_73/issue_73_1_fast_mode.py bugs/issue_75/issue_75_residues_Are_empty.py bugs/issue_75/issue_75_plot.py bugs/issue_76/test_ozone_unicode.py bugs/issue_79/test_ozone_missing_signal.py bugs/issue_80/test_ozone_missing_time.py bugs/issue_82/issue_82_long_cycles.py bugs/issue_94/issue_94.py  
 

cross_validation/test_air_passengers_cross_valid.py  :  
	 -$(PYTHON)  tests/cross_validation/test_air_passengers_cross_valid.py  >  logs/cross_validation_test_air_passengers_cross_valid.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/cross_validation/test_air_passengers_cross_valid.log logs/cross_validation_test_air_passengers_cross_valid.log  >  logs/cross_validation_test_air_passengers_cross_valid.log.diff
	 tail -10  logs/cross_validation_test_air_passengers_cross_valid.log.diff 

cross_validation/test_ozone_cross_valid.py  :  
	 -$(PYTHON)  tests/cross_validation/test_ozone_cross_valid.py  >  logs/cross_validation_test_ozone_cross_valid.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/cross_validation/test_ozone_cross_valid.log logs/cross_validation_test_ozone_cross_valid.log  >  logs/cross_validation_test_ozone_cross_valid.log.diff
	 tail -10  logs/cross_validation_test_ozone_cross_valid.log.diff 

cross_validation/test_ozone_custom_split.py  :  
	 -$(PYTHON)  tests/cross_validation/test_ozone_custom_split.py  >  logs/cross_validation_test_ozone_custom_split.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/cross_validation/test_ozone_custom_split.log logs/cross_validation_test_ozone_custom_split.log  >  logs/cross_validation_test_ozone_custom_split.log.diff
	 tail -10  logs/cross_validation_test_ozone_custom_split.log.diff 



 cross_validation :  cross_validation/test_ozone_custom_split.py cross_validation/test_ozone_cross_valid.py cross_validation/test_air_passengers_cross_valid.py  
 

croston/__init__.py  :  
	 -$(PYTHON)  tests/croston/__init__.py  >  logs/croston___init__.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/croston/__init__.log logs/croston___init__.log  >  logs/croston___init__.log.diff
	 tail -10  logs/croston___init__.log.diff 

croston/croston_test_1_None.py  :  
	 -$(PYTHON)  tests/croston/croston_test_1_None.py  >  logs/croston_croston_test_1_None.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/croston/croston_test_1_None.log logs/croston_croston_test_1_None.log  >  logs/croston_croston_test_1_None.log.diff
	 tail -10  logs/croston_croston_test_1_None.log.diff 

croston/croston_test_1_SBA.py  :  
	 -$(PYTHON)  tests/croston/croston_test_1_SBA.py  >  logs/croston_croston_test_1_SBA.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/croston/croston_test_1_SBA.log logs/croston_croston_test_1_SBA.log  >  logs/croston_croston_test_1_SBA.log.diff
	 tail -10  logs/croston_croston_test_1_SBA.log.diff 

croston/croston_test_1_SBA_linear_trend.py  :  
	 -$(PYTHON)  tests/croston/croston_test_1_SBA_linear_trend.py  >  logs/croston_croston_test_1_SBA_linear_trend.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/croston/croston_test_1_SBA_linear_trend.log logs/croston_croston_test_1_SBA_linear_trend.log  >  logs/croston_croston_test_1_SBA_linear_trend.log.diff
	 tail -10  logs/croston_croston_test_1_SBA_linear_trend.log.diff 

croston/croston_test_1_SBJ.py  :  
	 -$(PYTHON)  tests/croston/croston_test_1_SBJ.py  >  logs/croston_croston_test_1_SBJ.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/croston/croston_test_1_SBJ.log logs/croston_croston_test_1_SBJ.log  >  logs/croston_croston_test_1_SBJ.log.diff
	 tail -10  logs/croston_croston_test_1_SBJ.log.diff 

croston/croston_test_1_SBJ_linear_trend.py  :  
	 -$(PYTHON)  tests/croston/croston_test_1_SBJ_linear_trend.py  >  logs/croston_croston_test_1_SBJ_linear_trend.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/croston/croston_test_1_SBJ_linear_trend.log logs/croston_croston_test_1_SBJ_linear_trend.log  >  logs/croston_croston_test_1_SBJ_linear_trend.log.diff
	 tail -10  logs/croston_croston_test_1_SBJ_linear_trend.log.diff 

croston/croston_test_1_legacy_linear_trend.py  :  
	 -$(PYTHON)  tests/croston/croston_test_1_legacy_linear_trend.py  >  logs/croston_croston_test_1_legacy_linear_trend.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/croston/croston_test_1_legacy_linear_trend.log logs/croston_croston_test_1_legacy_linear_trend.log  >  logs/croston_croston_test_1_legacy_linear_trend.log.diff
	 tail -10  logs/croston_croston_test_1_legacy_linear_trend.log.diff 

croston/croston_tests.py  :  
	 -$(PYTHON)  tests/croston/croston_tests.py  >  logs/croston_croston_tests.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/croston/croston_tests.log logs/croston_croston_tests.log  >  logs/croston_croston_tests.log.diff
	 tail -10  logs/croston_croston_tests.log.diff 

croston/test_air_passengers_croston_only.py  :  
	 -$(PYTHON)  tests/croston/test_air_passengers_croston_only.py  >  logs/croston_test_air_passengers_croston_only.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/croston/test_air_passengers_croston_only.log logs/croston_test_air_passengers_croston_only.log  >  logs/croston_test_air_passengers_croston_only.log.diff
	 tail -10  logs/croston_test_air_passengers_croston_only.log.diff 

croston/test_croston_fpp2_counts_example.py  :  
	 -$(PYTHON)  tests/croston/test_croston_fpp2_counts_example.py  >  logs/croston_test_croston_fpp2_counts_example.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/croston/test_croston_fpp2_counts_example.log logs/croston_test_croston_fpp2_counts_example.log  >  logs/croston_test_croston_fpp2_counts_example.log.diff
	 tail -10  logs/croston_test_croston_fpp2_counts_example.log.diff 

croston/test_croston_fpp2_counts_example_optimize.py  :  
	 -$(PYTHON)  tests/croston/test_croston_fpp2_counts_example_optimize.py  >  logs/croston_test_croston_fpp2_counts_example_optimize.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/croston/test_croston_fpp2_counts_example_optimize.log logs/croston_test_croston_fpp2_counts_example_optimize.log  >  logs/croston_test_croston_fpp2_counts_example_optimize.log.diff
	 tail -10  logs/croston_test_croston_fpp2_counts_example_optimize.log.diff 

croston/test_ozone_croston_only.py  :  
	 -$(PYTHON)  tests/croston/test_ozone_croston_only.py  >  logs/croston_test_ozone_croston_only.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/croston/test_ozone_croston_only.log logs/croston_test_ozone_croston_only.log  >  logs/croston_test_ozone_croston_only.log.diff
	 tail -10  logs/croston_test_ozone_croston_only.log.diff 

croston/test_ozone_croston_only_2.py  :  
	 -$(PYTHON)  tests/croston/test_ozone_croston_only_2.py  >  logs/croston_test_ozone_croston_only_2.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/croston/test_ozone_croston_only_2.log logs/croston_test_ozone_croston_only_2.log  >  logs/croston_test_ozone_croston_only_2.log.diff
	 tail -10  logs/croston_test_ozone_croston_only_2.log.diff 



 croston :  croston/test_ozone_croston_only_2.py croston/test_ozone_croston_only.py croston/test_croston_fpp2_counts_example_optimize.py croston/test_croston_fpp2_counts_example.py croston/test_air_passengers_croston_only.py croston/croston_tests.py croston/croston_test_1_legacy_linear_trend.py croston/croston_test_1_SBJ_linear_trend.py croston/croston_test_1_SBJ.py croston/croston_test_1_SBA_linear_trend.py croston/croston_test_1_SBA.py croston/croston_test_1_None.py croston/__init__.py  
 

demos/demo1.py  :  
	 -$(PYTHON)  tests/demos/demo1.py  >  logs/demos_demo1.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/demos/demo1.log logs/demos_demo1.log  >  logs/demos_demo1.log.diff
	 tail -10  logs/demos_demo1.log.diff 



 demos :  demos/demo1.py  
 

exog/test_ozone_exogenous.py  :  
	 -$(PYTHON)  tests/exog/test_ozone_exogenous.py  >  logs/exog_test_ozone_exogenous.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/exog/test_ozone_exogenous.log logs/exog_test_ozone_exogenous.log  >  logs/exog_test_ozone_exogenous.log.diff
	 tail -10  logs/exog_test_ozone_exogenous.log.diff 

exog/test_ozone_exogenous_ARX.py  :  
	 -$(PYTHON)  tests/exog/test_ozone_exogenous_ARX.py  >  logs/exog_test_ozone_exogenous_ARX.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/exog/test_ozone_exogenous_ARX.log logs/exog_test_ozone_exogenous_ARX.log  >  logs/exog_test_ozone_exogenous_ARX.log.diff
	 tail -10  logs/exog_test_ozone_exogenous_ARX.log.diff 

exog/test_ozone_exogenous_LGBX.py  :  
	 -$(PYTHON)  tests/exog/test_ozone_exogenous_LGBX.py  >  logs/exog_test_ozone_exogenous_LGBX.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/exog/test_ozone_exogenous_LGBX.log logs/exog_test_ozone_exogenous_LGBX.log  >  logs/exog_test_ozone_exogenous_LGBX.log.diff
	 tail -10  logs/exog_test_ozone_exogenous_LGBX.log.diff 

exog/test_ozone_exogenous_LSTMX.py  :  
	 -$(PYTHON)  tests/exog/test_ozone_exogenous_LSTMX.py  >  logs/exog_test_ozone_exogenous_LSTMX.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/exog/test_ozone_exogenous_LSTMX.log logs/exog_test_ozone_exogenous_LSTMX.log  >  logs/exog_test_ozone_exogenous_LSTMX.log.diff
	 tail -10  logs/exog_test_ozone_exogenous_LSTMX.log.diff 

exog/test_ozone_exogenous_MLPX.py  :  
	 -$(PYTHON)  tests/exog/test_ozone_exogenous_MLPX.py  >  logs/exog_test_ozone_exogenous_MLPX.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/exog/test_ozone_exogenous_MLPX.log logs/exog_test_ozone_exogenous_MLPX.log  >  logs/exog_test_ozone_exogenous_MLPX.log.diff
	 tail -10  logs/exog_test_ozone_exogenous_MLPX.log.diff 

exog/test_ozone_exogenous_SVRX.py  :  
	 -$(PYTHON)  tests/exog/test_ozone_exogenous_SVRX.py  >  logs/exog_test_ozone_exogenous_SVRX.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/exog/test_ozone_exogenous_SVRX.log logs/exog_test_ozone_exogenous_SVRX.log  >  logs/exog_test_ozone_exogenous_SVRX.log.diff
	 tail -10  logs/exog_test_ozone_exogenous_SVRX.log.diff 

exog/test_ozone_exogenous_XGBX.py  :  
	 -$(PYTHON)  tests/exog/test_ozone_exogenous_XGBX.py  >  logs/exog_test_ozone_exogenous_XGBX.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/exog/test_ozone_exogenous_XGBX.log logs/exog_test_ozone_exogenous_XGBX.log  >  logs/exog_test_ozone_exogenous_XGBX.log.diff
	 tail -10  logs/exog_test_ozone_exogenous_XGBX.log.diff 

exog/test_ozone_exogenous_profile.py  :  
	 -$(PYTHON)  tests/exog/test_ozone_exogenous_profile.py  >  logs/exog_test_ozone_exogenous_profile.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/exog/test_ozone_exogenous_profile.log logs/exog_test_ozone_exogenous_profile.log  >  logs/exog_test_ozone_exogenous_profile.log.diff
	 tail -10  logs/exog_test_ozone_exogenous_profile.log.diff 

exog/test_ozone_exogenous_with_categorical.py  :  
	 -$(PYTHON)  tests/exog/test_ozone_exogenous_with_categorical.py  >  logs/exog_test_ozone_exogenous_with_categorical.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/exog/test_ozone_exogenous_with_categorical.log logs/exog_test_ozone_exogenous_with_categorical.log  >  logs/exog_test_ozone_exogenous_with_categorical.log.diff
	 tail -10  logs/exog_test_ozone_exogenous_with_categorical.log.diff 

exog/test_random_exogenous.py  :  
	 -$(PYTHON)  tests/exog/test_random_exogenous.py  >  logs/exog_test_random_exogenous.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/exog/test_random_exogenous.log logs/exog_test_random_exogenous.log  >  logs/exog_test_random_exogenous.log.diff
	 tail -10  logs/exog_test_random_exogenous.log.diff 



 exog :  exog/test_random_exogenous.py exog/test_ozone_exogenous_with_categorical.py exog/test_ozone_exogenous_profile.py exog/test_ozone_exogenous_XGBX.py exog/test_ozone_exogenous_SVRX.py exog/test_ozone_exogenous_MLPX.py exog/test_ozone_exogenous_LSTMX.py exog/test_ozone_exogenous_LGBX.py exog/test_ozone_exogenous_ARX.py exog/test_ozone_exogenous.py  
 

expsmooth/expsmooth_dataset_ausgdp.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_ausgdp.py  >  logs/expsmooth_expsmooth_dataset_ausgdp.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth/expsmooth_dataset_ausgdp.log logs/expsmooth_expsmooth_dataset_ausgdp.log  >  logs/expsmooth_expsmooth_dataset_ausgdp.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_ausgdp.log.diff 

expsmooth/expsmooth_dataset_bonds.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_bonds.py  >  logs/expsmooth_expsmooth_dataset_bonds.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth/expsmooth_dataset_bonds.log logs/expsmooth_expsmooth_dataset_bonds.log  >  logs/expsmooth_expsmooth_dataset_bonds.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_bonds.log.diff 

expsmooth/expsmooth_dataset_canadagas.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_canadagas.py  >  logs/expsmooth_expsmooth_dataset_canadagas.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth/expsmooth_dataset_canadagas.log logs/expsmooth_expsmooth_dataset_canadagas.log  >  logs/expsmooth_expsmooth_dataset_canadagas.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_canadagas.log.diff 

expsmooth/expsmooth_dataset_carparts.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_carparts.py  >  logs/expsmooth_expsmooth_dataset_carparts.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth/expsmooth_dataset_carparts.log logs/expsmooth_expsmooth_dataset_carparts.log  >  logs/expsmooth_expsmooth_dataset_carparts.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_carparts.log.diff 

expsmooth/expsmooth_dataset_dji.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_dji.py  >  logs/expsmooth_expsmooth_dataset_dji.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth/expsmooth_dataset_dji.log logs/expsmooth_expsmooth_dataset_dji.log  >  logs/expsmooth_expsmooth_dataset_dji.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_dji.log.diff 

expsmooth/expsmooth_dataset_djiclose.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_djiclose.py  >  logs/expsmooth_expsmooth_dataset_djiclose.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth/expsmooth_dataset_djiclose.log logs/expsmooth_expsmooth_dataset_djiclose.log  >  logs/expsmooth_expsmooth_dataset_djiclose.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_djiclose.log.diff 

expsmooth/expsmooth_dataset_enplanements.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_enplanements.py  >  logs/expsmooth_expsmooth_dataset_enplanements.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth/expsmooth_dataset_enplanements.log logs/expsmooth_expsmooth_dataset_enplanements.log  >  logs/expsmooth_expsmooth_dataset_enplanements.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_enplanements.log.diff 

expsmooth/expsmooth_dataset_fmsales.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_fmsales.py  >  logs/expsmooth_expsmooth_dataset_fmsales.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth/expsmooth_dataset_fmsales.log logs/expsmooth_expsmooth_dataset_fmsales.log  >  logs/expsmooth_expsmooth_dataset_fmsales.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_fmsales.log.diff 

expsmooth/expsmooth_dataset_freight.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_freight.py  >  logs/expsmooth_expsmooth_dataset_freight.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth/expsmooth_dataset_freight.log logs/expsmooth_expsmooth_dataset_freight.log  >  logs/expsmooth_expsmooth_dataset_freight.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_freight.log.diff 

expsmooth/expsmooth_dataset_frexport.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_frexport.py  >  logs/expsmooth_expsmooth_dataset_frexport.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth/expsmooth_dataset_frexport.log logs/expsmooth_expsmooth_dataset_frexport.log  >  logs/expsmooth_expsmooth_dataset_frexport.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_frexport.log.diff 

expsmooth/expsmooth_dataset_gasprice.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_gasprice.py  >  logs/expsmooth_expsmooth_dataset_gasprice.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth/expsmooth_dataset_gasprice.log logs/expsmooth_expsmooth_dataset_gasprice.log  >  logs/expsmooth_expsmooth_dataset_gasprice.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_gasprice.log.diff 

expsmooth/expsmooth_dataset_hospital.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_hospital.py  >  logs/expsmooth_expsmooth_dataset_hospital.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth/expsmooth_dataset_hospital.log logs/expsmooth_expsmooth_dataset_hospital.log  >  logs/expsmooth_expsmooth_dataset_hospital.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_hospital.log.diff 

expsmooth/expsmooth_dataset_jewelry.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_jewelry.py  >  logs/expsmooth_expsmooth_dataset_jewelry.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth/expsmooth_dataset_jewelry.log logs/expsmooth_expsmooth_dataset_jewelry.log  >  logs/expsmooth_expsmooth_dataset_jewelry.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_jewelry.log.diff 

expsmooth/expsmooth_dataset_mcopper.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_mcopper.py  >  logs/expsmooth_expsmooth_dataset_mcopper.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth/expsmooth_dataset_mcopper.log logs/expsmooth_expsmooth_dataset_mcopper.log  >  logs/expsmooth_expsmooth_dataset_mcopper.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_mcopper.log.diff 

expsmooth/expsmooth_dataset_msales.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_msales.py  >  logs/expsmooth_expsmooth_dataset_msales.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth/expsmooth_dataset_msales.log logs/expsmooth_expsmooth_dataset_msales.log  >  logs/expsmooth_expsmooth_dataset_msales.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_msales.log.diff 

expsmooth/expsmooth_dataset_partx.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_partx.py  >  logs/expsmooth_expsmooth_dataset_partx.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth/expsmooth_dataset_partx.log logs/expsmooth_expsmooth_dataset_partx.log  >  logs/expsmooth_expsmooth_dataset_partx.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_partx.log.diff 

expsmooth/expsmooth_dataset_test.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_test.py  >  logs/expsmooth_expsmooth_dataset_test.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth/expsmooth_dataset_test.log logs/expsmooth_expsmooth_dataset_test.log  >  logs/expsmooth_expsmooth_dataset_test.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_test.log.diff 

expsmooth/expsmooth_dataset_ukcars.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_ukcars.py  >  logs/expsmooth_expsmooth_dataset_ukcars.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth/expsmooth_dataset_ukcars.log logs/expsmooth_expsmooth_dataset_ukcars.log  >  logs/expsmooth_expsmooth_dataset_ukcars.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_ukcars.log.diff 

expsmooth/expsmooth_dataset_unemp.cci.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_unemp.cci.py  >  logs/expsmooth_expsmooth_dataset_unemp.cci.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth/expsmooth_dataset_unemp.cci.log logs/expsmooth_expsmooth_dataset_unemp.cci.log  >  logs/expsmooth_expsmooth_dataset_unemp.cci.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_unemp.cci.log.diff 

expsmooth/expsmooth_dataset_usgdp.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_usgdp.py  >  logs/expsmooth_expsmooth_dataset_usgdp.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth/expsmooth_dataset_usgdp.log logs/expsmooth_expsmooth_dataset_usgdp.log  >  logs/expsmooth_expsmooth_dataset_usgdp.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_usgdp.log.diff 

expsmooth/expsmooth_dataset_usnetelec.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_usnetelec.py  >  logs/expsmooth_expsmooth_dataset_usnetelec.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth/expsmooth_dataset_usnetelec.log logs/expsmooth_expsmooth_dataset_usnetelec.log  >  logs/expsmooth_expsmooth_dataset_usnetelec.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_usnetelec.log.diff 

expsmooth/expsmooth_dataset_utility.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_utility.py  >  logs/expsmooth_expsmooth_dataset_utility.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth/expsmooth_dataset_utility.log logs/expsmooth_expsmooth_dataset_utility.log  >  logs/expsmooth_expsmooth_dataset_utility.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_utility.log.diff 

expsmooth/expsmooth_dataset_vehicles.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_vehicles.py  >  logs/expsmooth_expsmooth_dataset_vehicles.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth/expsmooth_dataset_vehicles.log logs/expsmooth_expsmooth_dataset_vehicles.log  >  logs/expsmooth_expsmooth_dataset_vehicles.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_vehicles.log.diff 

expsmooth/expsmooth_dataset_visitors.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_visitors.py  >  logs/expsmooth_expsmooth_dataset_visitors.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth/expsmooth_dataset_visitors.log logs/expsmooth_expsmooth_dataset_visitors.log  >  logs/expsmooth_expsmooth_dataset_visitors.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_visitors.log.diff 

expsmooth/expsmooth_dataset_xrates.py  :  
	 -$(PYTHON)  tests/expsmooth/expsmooth_dataset_xrates.py  >  logs/expsmooth_expsmooth_dataset_xrates.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/expsmooth/expsmooth_dataset_xrates.log logs/expsmooth_expsmooth_dataset_xrates.log  >  logs/expsmooth_expsmooth_dataset_xrates.log.diff
	 tail -10  logs/expsmooth_expsmooth_dataset_xrates.log.diff 



 expsmooth :  expsmooth/expsmooth_dataset_xrates.py expsmooth/expsmooth_dataset_visitors.py expsmooth/expsmooth_dataset_vehicles.py expsmooth/expsmooth_dataset_utility.py expsmooth/expsmooth_dataset_usnetelec.py expsmooth/expsmooth_dataset_usgdp.py expsmooth/expsmooth_dataset_unemp.cci.py expsmooth/expsmooth_dataset_ukcars.py expsmooth/expsmooth_dataset_test.py expsmooth/expsmooth_dataset_partx.py expsmooth/expsmooth_dataset_msales.py expsmooth/expsmooth_dataset_mcopper.py expsmooth/expsmooth_dataset_jewelry.py expsmooth/expsmooth_dataset_hospital.py expsmooth/expsmooth_dataset_gasprice.py expsmooth/expsmooth_dataset_frexport.py expsmooth/expsmooth_dataset_freight.py expsmooth/expsmooth_dataset_fmsales.py expsmooth/expsmooth_dataset_enplanements.py expsmooth/expsmooth_dataset_djiclose.py expsmooth/expsmooth_dataset_dji.py expsmooth/expsmooth_dataset_carparts.py expsmooth/expsmooth_dataset_canadagas.py expsmooth/expsmooth_dataset_bonds.py expsmooth/expsmooth_dataset_ausgdp.py  
 

func/test_air_passengers.py  :  
	 -$(PYTHON)  tests/func/test_air_passengers.py  >  logs/func_test_air_passengers.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/func/test_air_passengers.log logs/func_test_air_passengers.log  >  logs/func_test_air_passengers.log.diff
	 tail -10  logs/func_test_air_passengers.log.diff 

func/test_air_passengers_no_vote.py  :  
	 -$(PYTHON)  tests/func/test_air_passengers_no_vote.py  >  logs/func_test_air_passengers_no_vote.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/func/test_air_passengers_no_vote.log logs/func_test_air_passengers_no_vote.log  >  logs/func_test_air_passengers_no_vote.log.diff
	 tail -10  logs/func_test_air_passengers_no_vote.log.diff 

func/test_ar.py  :  
	 -$(PYTHON)  tests/func/test_ar.py  >  logs/func_test_ar.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/func/test_ar.log logs/func_test_ar.log  >  logs/func_test_ar.log.diff
	 tail -10  logs/func_test_ar.log.diff 

func/test_const_signal.py  :  
	 -$(PYTHON)  tests/func/test_const_signal.py  >  logs/func_test_const_signal.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/func/test_const_signal.log logs/func_test_const_signal.log  >  logs/func_test_const_signal.log.diff
	 tail -10  logs/func_test_const_signal.log.diff 

func/test_cycle.py  :  
	 -$(PYTHON)  tests/func/test_cycle.py  >  logs/func_test_cycle.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/func/test_cycle.log logs/func_test_cycle.log  >  logs/func_test_cycle.log.diff
	 tail -10  logs/func_test_cycle.log.diff 

func/test_cycles_full.py  :  
	 -$(PYTHON)  tests/func/test_cycles_full.py  >  logs/func_test_cycles_full.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/func/test_cycles_full.log logs/func_test_cycles_full.log  >  logs/func_test_cycles_full.log.diff
	 tail -10  logs/func_test_cycles_full.log.diff 

func/test_final_touch_1.py  :  
	 -$(PYTHON)  tests/func/test_final_touch_1.py  >  logs/func_test_final_touch_1.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/func/test_final_touch_1.log logs/func_test_final_touch_1.log  >  logs/func_test_final_touch_1.log.diff
	 tail -10  logs/func_test_final_touch_1.log.diff 

func/test_final_touch_2.py  :  
	 -$(PYTHON)  tests/func/test_final_touch_2.py  >  logs/func_test_final_touch_2.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/func/test_final_touch_2.log logs/func_test_final_touch_2.log  >  logs/func_test_final_touch_2.log.diff
	 tail -10  logs/func_test_final_touch_2.log.diff 

func/test_ozone.py  :  
	 -$(PYTHON)  tests/func/test_ozone.py  >  logs/func_test_ozone.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/func/test_ozone.log logs/func_test_ozone.log  >  logs/func_test_ozone.log.diff
	 tail -10  logs/func_test_ozone.log.diff 

func/test_ozone_bench_mode.py  :  
	 -$(PYTHON)  tests/func/test_ozone_bench_mode.py  >  logs/func_test_ozone_bench_mode.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/func/test_ozone_bench_mode.log logs/func_test_ozone_bench_mode.log  >  logs/func_test_ozone_bench_mode.log.diff
	 tail -10  logs/func_test_ozone_bench_mode.log.diff 

func/test_ozone_disable_seasonal.py  :  
	 -$(PYTHON)  tests/func/test_ozone_disable_seasonal.py  >  logs/func_test_ozone_disable_seasonal.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/func/test_ozone_disable_seasonal.log logs/func_test_ozone_disable_seasonal.log  >  logs/func_test_ozone_disable_seasonal.log.diff
	 tail -10  logs/func_test_ozone_disable_seasonal.log.diff 

func/test_ozone_horizon_one.py  :  
	 -$(PYTHON)  tests/func/test_ozone_horizon_one.py  >  logs/func_test_ozone_horizon_one.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/func/test_ozone_horizon_one.log logs/func_test_ozone_horizon_one.log  >  logs/func_test_ozone_horizon_one.log.diff
	 tail -10  logs/func_test_ozone_horizon_one.log.diff 

func/test_ozone_no_vote.py  :  
	 -$(PYTHON)  tests/func/test_ozone_no_vote.py  >  logs/func_test_ozone_no_vote.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/func/test_ozone_no_vote.log logs/func_test_ozone_no_vote.log  >  logs/func_test_ozone_no_vote.log.diff
	 tail -10  logs/func_test_ozone_no_vote.log.diff 

func/test_ozone_with_logging.py  :  
	 -$(PYTHON)  tests/func/test_ozone_with_logging.py  >  logs/func_test_ozone_with_logging.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/func/test_ozone_with_logging.log logs/func_test_ozone_with_logging.log  >  logs/func_test_ozone_with_logging.log.diff
	 tail -10  logs/func_test_ozone_with_logging.log.diff 

func/test_version.py  :  
	 -$(PYTHON)  tests/func/test_version.py  >  logs/func_test_version.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/func/test_version.log logs/func_test_version.log  >  logs/func_test_version.log.diff
	 tail -10  logs/func_test_version.log.diff 



 func :  func/test_version.py func/test_ozone_with_logging.py func/test_ozone_no_vote.py func/test_ozone_horizon_one.py func/test_ozone_disable_seasonal.py func/test_ozone_bench_mode.py func/test_ozone.py func/test_final_touch_2.py func/test_final_touch_1.py func/test_cycles_full.py func/test_cycle.py func/test_const_signal.py func/test_ar.py func/test_air_passengers_no_vote.py func/test_air_passengers.py  
 

heroku/build_generic_heroku_model.py  :  
	 -$(PYTHON)  tests/heroku/build_generic_heroku_model.py  >  logs/heroku_build_generic_heroku_model.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/heroku/build_generic_heroku_model.log logs/heroku_build_generic_heroku_model.log  >  logs/heroku_build_generic_heroku_model.log.diff
	 tail -10  logs/heroku_build_generic_heroku_model.log.diff 

heroku/test_air_passengers_heroku.py  :  
	 -$(PYTHON)  tests/heroku/test_air_passengers_heroku.py  >  logs/heroku_test_air_passengers_heroku.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/heroku/test_air_passengers_heroku.log logs/heroku_test_air_passengers_heroku.log  >  logs/heroku_test_air_passengers_heroku.log.diff
	 tail -10  logs/heroku_test_air_passengers_heroku.log.diff 

heroku/test_ozone_exog_heroku.py  :  
	 -$(PYTHON)  tests/heroku/test_ozone_exog_heroku.py  >  logs/heroku_test_ozone_exog_heroku.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/heroku/test_ozone_exog_heroku.log logs/heroku_test_ozone_exog_heroku.log  >  logs/heroku_test_ozone_exog_heroku.log.diff
	 tail -10  logs/heroku_test_ozone_exog_heroku.log.diff 

heroku/test_ozone_heroku.py  :  
	 -$(PYTHON)  tests/heroku/test_ozone_heroku.py  >  logs/heroku_test_ozone_heroku.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/heroku/test_ozone_heroku.log logs/heroku_test_ozone_heroku.log  >  logs/heroku_test_ozone_heroku.log.diff
	 tail -10  logs/heroku_test_ozone_heroku.log.diff 

heroku/test_seasonal_week_of_year.py  :  
	 -$(PYTHON)  tests/heroku/test_seasonal_week_of_year.py  >  logs/heroku_test_seasonal_week_of_year.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/heroku/test_seasonal_week_of_year.log logs/heroku_test_seasonal_week_of_year.log  >  logs/heroku_test_seasonal_week_of_year.log.diff
	 tail -10  logs/heroku_test_seasonal_week_of_year.log.diff 



 heroku :  heroku/test_seasonal_week_of_year.py heroku/test_ozone_heroku.py heroku/test_ozone_exog_heroku.py heroku/test_air_passengers_heroku.py heroku/build_generic_heroku_model.py  
 

hierarchical/test_grouped.py  :  
	 -$(PYTHON)  tests/hierarchical/test_grouped.py  >  logs/hierarchical_test_grouped.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical/test_grouped.log logs/hierarchical_test_grouped.log  >  logs/hierarchical_test_grouped.log.diff
	 tail -10  logs/hierarchical_test_grouped.log.diff 

hierarchical/test_grouped_signals_AllMethods.py  :  
	 -$(PYTHON)  tests/hierarchical/test_grouped_signals_AllMethods.py  >  logs/hierarchical_test_grouped_signals_AllMethods.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical/test_grouped_signals_AllMethods.log logs/hierarchical_test_grouped_signals_AllMethods.log  >  logs/hierarchical_test_grouped_signals_AllMethods.log.diff
	 tail -10  logs/hierarchical_test_grouped_signals_AllMethods.log.diff 

hierarchical/test_grouped_signals_AllMethods_2.py  :  
	 -$(PYTHON)  tests/hierarchical/test_grouped_signals_AllMethods_2.py  >  logs/hierarchical_test_grouped_signals_AllMethods_2.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical/test_grouped_signals_AllMethods_2.log logs/hierarchical_test_grouped_signals_AllMethods_2.log  >  logs/hierarchical_test_grouped_signals_AllMethods_2.log.diff
	 tail -10  logs/hierarchical_test_grouped_signals_AllMethods_2.log.diff 

hierarchical/test_grouped_signals_AllMethods_Exogenous_all_nodes.py  :  
	 -$(PYTHON)  tests/hierarchical/test_grouped_signals_AllMethods_Exogenous_all_nodes.py  >  logs/hierarchical_test_grouped_signals_AllMethods_Exogenous_all_nodes.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical/test_grouped_signals_AllMethods_Exogenous_all_nodes.log logs/hierarchical_test_grouped_signals_AllMethods_Exogenous_all_nodes.log  >  logs/hierarchical_test_grouped_signals_AllMethods_Exogenous_all_nodes.log.diff
	 tail -10  logs/hierarchical_test_grouped_signals_AllMethods_Exogenous_all_nodes.log.diff 

hierarchical/test_grouped_signals_AllMethods_Exogenous_all_nodes_force_ARX.py  :  
	 -$(PYTHON)  tests/hierarchical/test_grouped_signals_AllMethods_Exogenous_all_nodes_force_ARX.py  >  logs/hierarchical_test_grouped_signals_AllMethods_Exogenous_all_nodes_force_ARX.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical/test_grouped_signals_AllMethods_Exogenous_all_nodes_force_ARX.log logs/hierarchical_test_grouped_signals_AllMethods_Exogenous_all_nodes_force_ARX.log  >  logs/hierarchical_test_grouped_signals_AllMethods_Exogenous_all_nodes_force_ARX.log.diff
	 tail -10  logs/hierarchical_test_grouped_signals_AllMethods_Exogenous_all_nodes_force_ARX.log.diff 

hierarchical/test_grouped_signals_AllMethods_Exogenous_all_nodes_force_XGBX.py  :  
	 -$(PYTHON)  tests/hierarchical/test_grouped_signals_AllMethods_Exogenous_all_nodes_force_XGBX.py  >  logs/hierarchical_test_grouped_signals_AllMethods_Exogenous_all_nodes_force_XGBX.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical/test_grouped_signals_AllMethods_Exogenous_all_nodes_force_XGBX.log logs/hierarchical_test_grouped_signals_AllMethods_Exogenous_all_nodes_force_XGBX.log  >  logs/hierarchical_test_grouped_signals_AllMethods_Exogenous_all_nodes_force_XGBX.log.diff
	 tail -10  logs/hierarchical_test_grouped_signals_AllMethods_Exogenous_all_nodes_force_XGBX.log.diff 

hierarchical/test_grouped_signals_AllMethods_Exogenous_per_node.py  :  
	 -$(PYTHON)  tests/hierarchical/test_grouped_signals_AllMethods_Exogenous_per_node.py  >  logs/hierarchical_test_grouped_signals_AllMethods_Exogenous_per_node.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical/test_grouped_signals_AllMethods_Exogenous_per_node.log logs/hierarchical_test_grouped_signals_AllMethods_Exogenous_per_node.log  >  logs/hierarchical_test_grouped_signals_AllMethods_Exogenous_per_node.log.diff
	 tail -10  logs/hierarchical_test_grouped_signals_AllMethods_Exogenous_per_node.log.diff 

hierarchical/test_grouped_signals_AllMethods_Exogenous_per_node_force_ARX.py  :  
	 -$(PYTHON)  tests/hierarchical/test_grouped_signals_AllMethods_Exogenous_per_node_force_ARX.py  >  logs/hierarchical_test_grouped_signals_AllMethods_Exogenous_per_node_force_ARX.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical/test_grouped_signals_AllMethods_Exogenous_per_node_force_ARX.log logs/hierarchical_test_grouped_signals_AllMethods_Exogenous_per_node_force_ARX.log  >  logs/hierarchical_test_grouped_signals_AllMethods_Exogenous_per_node_force_ARX.log.diff
	 tail -10  logs/hierarchical_test_grouped_signals_AllMethods_Exogenous_per_node_force_ARX.log.diff 

hierarchical/test_grouped_signals_AllMethods_Exogenous_per_node_force_XGBX.py  :  
	 -$(PYTHON)  tests/hierarchical/test_grouped_signals_AllMethods_Exogenous_per_node_force_XGBX.py  >  logs/hierarchical_test_grouped_signals_AllMethods_Exogenous_per_node_force_XGBX.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical/test_grouped_signals_AllMethods_Exogenous_per_node_force_XGBX.log logs/hierarchical_test_grouped_signals_AllMethods_Exogenous_per_node_force_XGBX.log  >  logs/hierarchical_test_grouped_signals_AllMethods_Exogenous_per_node_force_XGBX.log.diff
	 tail -10  logs/hierarchical_test_grouped_signals_AllMethods_Exogenous_per_node_force_XGBX.log.diff 

hierarchical/test_grouped_signals_BU.py  :  
	 -$(PYTHON)  tests/hierarchical/test_grouped_signals_BU.py  >  logs/hierarchical_test_grouped_signals_BU.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical/test_grouped_signals_BU.log logs/hierarchical_test_grouped_signals_BU.log  >  logs/hierarchical_test_grouped_signals_BU.log.diff
	 tail -10  logs/hierarchical_test_grouped_signals_BU.log.diff 

hierarchical/test_grouped_signals_MO.py  :  
	 -$(PYTHON)  tests/hierarchical/test_grouped_signals_MO.py  >  logs/hierarchical_test_grouped_signals_MO.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical/test_grouped_signals_MO.log logs/hierarchical_test_grouped_signals_MO.log  >  logs/hierarchical_test_grouped_signals_MO.log.diff
	 tail -10  logs/hierarchical_test_grouped_signals_MO.log.diff 

hierarchical/test_grouped_signals_OC.py  :  
	 -$(PYTHON)  tests/hierarchical/test_grouped_signals_OC.py  >  logs/hierarchical_test_grouped_signals_OC.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical/test_grouped_signals_OC.log logs/hierarchical_test_grouped_signals_OC.log  >  logs/hierarchical_test_grouped_signals_OC.log.diff
	 tail -10  logs/hierarchical_test_grouped_signals_OC.log.diff 

hierarchical/test_grouped_signals_TD.py  :  
	 -$(PYTHON)  tests/hierarchical/test_grouped_signals_TD.py  >  logs/hierarchical_test_grouped_signals_TD.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical/test_grouped_signals_TD.log logs/hierarchical_test_grouped_signals_TD.log  >  logs/hierarchical_test_grouped_signals_TD.log.diff
	 tail -10  logs/hierarchical_test_grouped_signals_TD.log.diff 

hierarchical/test_hierarchy_AU.py  :  
	 -$(PYTHON)  tests/hierarchical/test_hierarchy_AU.py  >  logs/hierarchical_test_hierarchy_AU.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical/test_hierarchy_AU.log logs/hierarchical_test_hierarchy_AU.log  >  logs/hierarchical_test_hierarchy_AU.log.diff
	 tail -10  logs/hierarchical_test_hierarchy_AU.log.diff 

hierarchical/test_hierarchy_AU_AllMethods.py  :  
	 -$(PYTHON)  tests/hierarchical/test_hierarchy_AU_AllMethods.py  >  logs/hierarchical_test_hierarchy_AU_AllMethods.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical/test_hierarchy_AU_AllMethods.log logs/hierarchical_test_hierarchy_AU_AllMethods.log  >  logs/hierarchical_test_hierarchy_AU_AllMethods.log.diff
	 tail -10  logs/hierarchical_test_hierarchy_AU_AllMethods.log.diff 

hierarchical/test_hierarchy_AU_AllMethods_Exogenous_all_nodes.py  :  
	 -$(PYTHON)  tests/hierarchical/test_hierarchy_AU_AllMethods_Exogenous_all_nodes.py  >  logs/hierarchical_test_hierarchy_AU_AllMethods_Exogenous_all_nodes.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical/test_hierarchy_AU_AllMethods_Exogenous_all_nodes.log logs/hierarchical_test_hierarchy_AU_AllMethods_Exogenous_all_nodes.log  >  logs/hierarchical_test_hierarchy_AU_AllMethods_Exogenous_all_nodes.log.diff
	 tail -10  logs/hierarchical_test_hierarchy_AU_AllMethods_Exogenous_all_nodes.log.diff 

hierarchical/test_hierarchy_AU_AllMethods_Exogenous_per_node.py  :  
	 -$(PYTHON)  tests/hierarchical/test_hierarchy_AU_AllMethods_Exogenous_per_node.py  >  logs/hierarchical_test_hierarchy_AU_AllMethods_Exogenous_per_node.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical/test_hierarchy_AU_AllMethods_Exogenous_per_node.log logs/hierarchical_test_hierarchy_AU_AllMethods_Exogenous_per_node.log  >  logs/hierarchical_test_hierarchy_AU_AllMethods_Exogenous_per_node.log.diff
	 tail -10  logs/hierarchical_test_hierarchy_AU_AllMethods_Exogenous_per_node.log.diff 

hierarchical/test_hierarchy_AU_BU.py  :  
	 -$(PYTHON)  tests/hierarchical/test_hierarchy_AU_BU.py  >  logs/hierarchical_test_hierarchy_AU_BU.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical/test_hierarchy_AU_BU.log logs/hierarchical_test_hierarchy_AU_BU.log  >  logs/hierarchical_test_hierarchy_AU_BU.log.diff
	 tail -10  logs/hierarchical_test_hierarchy_AU_BU.log.diff 

hierarchical/test_hierarchy_AU_MO.py  :  
	 -$(PYTHON)  tests/hierarchical/test_hierarchy_AU_MO.py  >  logs/hierarchical_test_hierarchy_AU_MO.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical/test_hierarchy_AU_MO.log logs/hierarchical_test_hierarchy_AU_MO.log  >  logs/hierarchical_test_hierarchy_AU_MO.log.diff
	 tail -10  logs/hierarchical_test_hierarchy_AU_MO.log.diff 

hierarchical/test_hierarchy_AU_OC.py  :  
	 -$(PYTHON)  tests/hierarchical/test_hierarchy_AU_OC.py  >  logs/hierarchical_test_hierarchy_AU_OC.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical/test_hierarchy_AU_OC.log logs/hierarchical_test_hierarchy_AU_OC.log  >  logs/hierarchical_test_hierarchy_AU_OC.log.diff
	 tail -10  logs/hierarchical_test_hierarchy_AU_OC.log.diff 

hierarchical/test_hierarchy_AU_TD.py  :  
	 -$(PYTHON)  tests/hierarchical/test_hierarchy_AU_TD.py  >  logs/hierarchical_test_hierarchy_AU_TD.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/hierarchical/test_hierarchy_AU_TD.log logs/hierarchical_test_hierarchy_AU_TD.log  >  logs/hierarchical_test_hierarchy_AU_TD.log.diff
	 tail -10  logs/hierarchical_test_hierarchy_AU_TD.log.diff 



 hierarchical :  hierarchical/test_hierarchy_AU_TD.py hierarchical/test_hierarchy_AU_OC.py hierarchical/test_hierarchy_AU_MO.py hierarchical/test_hierarchy_AU_BU.py hierarchical/test_hierarchy_AU_AllMethods_Exogenous_per_node.py hierarchical/test_hierarchy_AU_AllMethods_Exogenous_all_nodes.py hierarchical/test_hierarchy_AU_AllMethods.py hierarchical/test_hierarchy_AU.py hierarchical/test_grouped_signals_TD.py hierarchical/test_grouped_signals_OC.py hierarchical/test_grouped_signals_MO.py hierarchical/test_grouped_signals_BU.py hierarchical/test_grouped_signals_AllMethods_Exogenous_per_node_force_XGBX.py hierarchical/test_grouped_signals_AllMethods_Exogenous_per_node_force_ARX.py hierarchical/test_grouped_signals_AllMethods_Exogenous_per_node.py hierarchical/test_grouped_signals_AllMethods_Exogenous_all_nodes_force_XGBX.py hierarchical/test_grouped_signals_AllMethods_Exogenous_all_nodes_force_ARX.py hierarchical/test_grouped_signals_AllMethods_Exogenous_all_nodes.py hierarchical/test_grouped_signals_AllMethods_2.py hierarchical/test_grouped_signals_AllMethods.py hierarchical/test_grouped.py  
 

individual_components/test_air_passengers_all_autoreg.py  :  
	 -$(PYTHON)  tests/individual_components/test_air_passengers_all_autoreg.py  >  logs/individual_components_test_air_passengers_all_autoreg.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/individual_components/test_air_passengers_all_autoreg.log logs/individual_components_test_air_passengers_all_autoreg.log  >  logs/individual_components_test_air_passengers_all_autoreg.log.diff
	 tail -10  logs/individual_components_test_air_passengers_all_autoreg.log.diff 

individual_components/test_air_passengers_all_decomposition_types.py  :  
	 -$(PYTHON)  tests/individual_components/test_air_passengers_all_decomposition_types.py  >  logs/individual_components_test_air_passengers_all_decomposition_types.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/individual_components/test_air_passengers_all_decomposition_types.log logs/individual_components_test_air_passengers_all_decomposition_types.log  >  logs/individual_components_test_air_passengers_all_decomposition_types.log.diff
	 tail -10  logs/individual_components_test_air_passengers_all_decomposition_types.log.diff 

individual_components/test_air_passengers_all_transforms.py  :  
	 -$(PYTHON)  tests/individual_components/test_air_passengers_all_transforms.py  >  logs/individual_components_test_air_passengers_all_transforms.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/individual_components/test_air_passengers_all_transforms.log logs/individual_components_test_air_passengers_all_transforms.log  >  logs/individual_components_test_air_passengers_all_transforms.log.diff
	 tail -10  logs/individual_components_test_air_passengers_all_transforms.log.diff 

individual_components/test_air_passengers_all_trends.py  :  
	 -$(PYTHON)  tests/individual_components/test_air_passengers_all_trends.py  >  logs/individual_components_test_air_passengers_all_trends.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/individual_components/test_air_passengers_all_trends.log logs/individual_components_test_air_passengers_all_trends.log  >  logs/individual_components_test_air_passengers_all_trends.log.diff
	 tail -10  logs/individual_components_test_air_passengers_all_trends.log.diff 

individual_components/test_ozone_all_autoreg.py  :  
	 -$(PYTHON)  tests/individual_components/test_ozone_all_autoreg.py  >  logs/individual_components_test_ozone_all_autoreg.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/individual_components/test_ozone_all_autoreg.log logs/individual_components_test_ozone_all_autoreg.log  >  logs/individual_components_test_ozone_all_autoreg.log.diff
	 tail -10  logs/individual_components_test_ozone_all_autoreg.log.diff 

individual_components/test_ozone_all_decomposition_types.py  :  
	 -$(PYTHON)  tests/individual_components/test_ozone_all_decomposition_types.py  >  logs/individual_components_test_ozone_all_decomposition_types.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/individual_components/test_ozone_all_decomposition_types.log logs/individual_components_test_ozone_all_decomposition_types.log  >  logs/individual_components_test_ozone_all_decomposition_types.log.diff
	 tail -10  logs/individual_components_test_ozone_all_decomposition_types.log.diff 

individual_components/test_ozone_all_transforms.py  :  
	 -$(PYTHON)  tests/individual_components/test_ozone_all_transforms.py  >  logs/individual_components_test_ozone_all_transforms.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/individual_components/test_ozone_all_transforms.log logs/individual_components_test_ozone_all_transforms.log  >  logs/individual_components_test_ozone_all_transforms.log.diff
	 tail -10  logs/individual_components_test_ozone_all_transforms.log.diff 

individual_components/test_ozone_all_trends.py  :  
	 -$(PYTHON)  tests/individual_components/test_ozone_all_trends.py  >  logs/individual_components_test_ozone_all_trends.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/individual_components/test_ozone_all_trends.log logs/individual_components_test_ozone_all_trends.log  >  logs/individual_components_test_ozone_all_trends.log.diff
	 tail -10  logs/individual_components_test_ozone_all_trends.log.diff 



 individual_components :  individual_components/test_ozone_all_trends.py individual_components/test_ozone_all_transforms.py individual_components/test_ozone_all_decomposition_types.py individual_components/test_ozone_all_autoreg.py individual_components/test_air_passengers_all_trends.py individual_components/test_air_passengers_all_transforms.py individual_components/test_air_passengers_all_decomposition_types.py individual_components/test_air_passengers_all_autoreg.py  
 

lgbm/test_air_passengers_lgb.py  :  
	 -$(PYTHON)  tests/lgbm/test_air_passengers_lgb.py  >  logs/lgbm_test_air_passengers_lgb.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/lgbm/test_air_passengers_lgb.log logs/lgbm_test_air_passengers_lgb.log  >  logs/lgbm_test_air_passengers_lgb.log.diff
	 tail -10  logs/lgbm_test_air_passengers_lgb.log.diff 

lgbm/test_air_passengers_lgb_only.py  :  
	 -$(PYTHON)  tests/lgbm/test_air_passengers_lgb_only.py  >  logs/lgbm_test_air_passengers_lgb_only.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/lgbm/test_air_passengers_lgb_only.log logs/lgbm_test_air_passengers_lgb_only.log  >  logs/lgbm_test_air_passengers_lgb_only.log.diff
	 tail -10  logs/lgbm_test_air_passengers_lgb_only.log.diff 

lgbm/test_air_passengers_lgb_only_with_custom_options.py  :  
	 -$(PYTHON)  tests/lgbm/test_air_passengers_lgb_only_with_custom_options.py  >  logs/lgbm_test_air_passengers_lgb_only_with_custom_options.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/lgbm/test_air_passengers_lgb_only_with_custom_options.log logs/lgbm_test_air_passengers_lgb_only_with_custom_options.log  >  logs/lgbm_test_air_passengers_lgb_only_with_custom_options.log.diff
	 tail -10  logs/lgbm_test_air_passengers_lgb_only_with_custom_options.log.diff 

lgbm/test_ozone_lgb.py  :  
	 -$(PYTHON)  tests/lgbm/test_ozone_lgb.py  >  logs/lgbm_test_ozone_lgb.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/lgbm/test_ozone_lgb.log logs/lgbm_test_ozone_lgb.log  >  logs/lgbm_test_ozone_lgb.log.diff
	 tail -10  logs/lgbm_test_ozone_lgb.log.diff 

lgbm/test_ozone_lgb_exogenous.py  :  
	 -$(PYTHON)  tests/lgbm/test_ozone_lgb_exogenous.py  >  logs/lgbm_test_ozone_lgb_exogenous.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/lgbm/test_ozone_lgb_exogenous.log logs/lgbm_test_ozone_lgb_exogenous.log  >  logs/lgbm_test_ozone_lgb_exogenous.log.diff
	 tail -10  logs/lgbm_test_ozone_lgb_exogenous.log.diff 

lgbm/test_ozone_lgb_only.py  :  
	 -$(PYTHON)  tests/lgbm/test_ozone_lgb_only.py  >  logs/lgbm_test_ozone_lgb_only.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/lgbm/test_ozone_lgb_only.log logs/lgbm_test_ozone_lgb_only.log  >  logs/lgbm_test_ozone_lgb_only.log.diff
	 tail -10  logs/lgbm_test_ozone_lgb_only.log.diff 

lgbm/test_ozone_lgbx_exogenous.py  :  
	 -$(PYTHON)  tests/lgbm/test_ozone_lgbx_exogenous.py  >  logs/lgbm_test_ozone_lgbx_exogenous.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/lgbm/test_ozone_lgbx_exogenous.log logs/lgbm_test_ozone_lgbx_exogenous.log  >  logs/lgbm_test_ozone_lgbx_exogenous.log.diff
	 tail -10  logs/lgbm_test_ozone_lgbx_exogenous.log.diff 



 lgbm :  lgbm/test_ozone_lgbx_exogenous.py lgbm/test_ozone_lgb_only.py lgbm/test_ozone_lgb_exogenous.py lgbm/test_ozone_lgb.py lgbm/test_air_passengers_lgb_only_with_custom_options.py lgbm/test_air_passengers_lgb_only.py lgbm/test_air_passengers_lgb.py  
 

missing_data/gen_air_passengers_tests.py  :  
	 -$(PYTHON)  tests/missing_data/gen_air_passengers_tests.py  >  logs/missing_data_gen_air_passengers_tests.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/gen_air_passengers_tests.log logs/missing_data_gen_air_passengers_tests.log  >  logs/missing_data_gen_air_passengers_tests.log.diff
	 tail -10  logs/missing_data_gen_air_passengers_tests.log.diff 

missing_data/gen_ozone_tests.py  :  
	 -$(PYTHON)  tests/missing_data/gen_ozone_tests.py  >  logs/missing_data_gen_ozone_tests.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/gen_ozone_tests.log logs/missing_data_gen_ozone_tests.log  >  logs/missing_data_gen_ozone_tests.log.diff
	 tail -10  logs/missing_data_gen_ozone_tests.log.diff 

missing_data/test_missing_data_air_passengers_DiscardRow_Constant.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_air_passengers_DiscardRow_Constant.py  >  logs/missing_data_test_missing_data_air_passengers_DiscardRow_Constant.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_air_passengers_DiscardRow_Constant.log logs/missing_data_test_missing_data_air_passengers_DiscardRow_Constant.log  >  logs/missing_data_test_missing_data_air_passengers_DiscardRow_Constant.log.diff
	 tail -10  logs/missing_data_test_missing_data_air_passengers_DiscardRow_Constant.log.diff 

missing_data/test_missing_data_air_passengers_DiscardRow_DiscardRow.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_air_passengers_DiscardRow_DiscardRow.py  >  logs/missing_data_test_missing_data_air_passengers_DiscardRow_DiscardRow.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_air_passengers_DiscardRow_DiscardRow.log logs/missing_data_test_missing_data_air_passengers_DiscardRow_DiscardRow.log  >  logs/missing_data_test_missing_data_air_passengers_DiscardRow_DiscardRow.log.diff
	 tail -10  logs/missing_data_test_missing_data_air_passengers_DiscardRow_DiscardRow.log.diff 

missing_data/test_missing_data_air_passengers_DiscardRow_Interpolate.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_air_passengers_DiscardRow_Interpolate.py  >  logs/missing_data_test_missing_data_air_passengers_DiscardRow_Interpolate.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_air_passengers_DiscardRow_Interpolate.log logs/missing_data_test_missing_data_air_passengers_DiscardRow_Interpolate.log  >  logs/missing_data_test_missing_data_air_passengers_DiscardRow_Interpolate.log.diff
	 tail -10  logs/missing_data_test_missing_data_air_passengers_DiscardRow_Interpolate.log.diff 

missing_data/test_missing_data_air_passengers_DiscardRow_Mean.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_air_passengers_DiscardRow_Mean.py  >  logs/missing_data_test_missing_data_air_passengers_DiscardRow_Mean.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_air_passengers_DiscardRow_Mean.log logs/missing_data_test_missing_data_air_passengers_DiscardRow_Mean.log  >  logs/missing_data_test_missing_data_air_passengers_DiscardRow_Mean.log.diff
	 tail -10  logs/missing_data_test_missing_data_air_passengers_DiscardRow_Mean.log.diff 

missing_data/test_missing_data_air_passengers_DiscardRow_Median.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_air_passengers_DiscardRow_Median.py  >  logs/missing_data_test_missing_data_air_passengers_DiscardRow_Median.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_air_passengers_DiscardRow_Median.log logs/missing_data_test_missing_data_air_passengers_DiscardRow_Median.log  >  logs/missing_data_test_missing_data_air_passengers_DiscardRow_Median.log.diff
	 tail -10  logs/missing_data_test_missing_data_air_passengers_DiscardRow_Median.log.diff 

missing_data/test_missing_data_air_passengers_DiscardRow_None.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_air_passengers_DiscardRow_None.py  >  logs/missing_data_test_missing_data_air_passengers_DiscardRow_None.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_air_passengers_DiscardRow_None.log logs/missing_data_test_missing_data_air_passengers_DiscardRow_None.log  >  logs/missing_data_test_missing_data_air_passengers_DiscardRow_None.log.diff
	 tail -10  logs/missing_data_test_missing_data_air_passengers_DiscardRow_None.log.diff 

missing_data/test_missing_data_air_passengers_DiscardRow_PreviousValue.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_air_passengers_DiscardRow_PreviousValue.py  >  logs/missing_data_test_missing_data_air_passengers_DiscardRow_PreviousValue.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_air_passengers_DiscardRow_PreviousValue.log logs/missing_data_test_missing_data_air_passengers_DiscardRow_PreviousValue.log  >  logs/missing_data_test_missing_data_air_passengers_DiscardRow_PreviousValue.log.diff
	 tail -10  logs/missing_data_test_missing_data_air_passengers_DiscardRow_PreviousValue.log.diff 

missing_data/test_missing_data_air_passengers_Interpolate_Constant.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_air_passengers_Interpolate_Constant.py  >  logs/missing_data_test_missing_data_air_passengers_Interpolate_Constant.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_air_passengers_Interpolate_Constant.log logs/missing_data_test_missing_data_air_passengers_Interpolate_Constant.log  >  logs/missing_data_test_missing_data_air_passengers_Interpolate_Constant.log.diff
	 tail -10  logs/missing_data_test_missing_data_air_passengers_Interpolate_Constant.log.diff 

missing_data/test_missing_data_air_passengers_Interpolate_DiscardRow.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_air_passengers_Interpolate_DiscardRow.py  >  logs/missing_data_test_missing_data_air_passengers_Interpolate_DiscardRow.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_air_passengers_Interpolate_DiscardRow.log logs/missing_data_test_missing_data_air_passengers_Interpolate_DiscardRow.log  >  logs/missing_data_test_missing_data_air_passengers_Interpolate_DiscardRow.log.diff
	 tail -10  logs/missing_data_test_missing_data_air_passengers_Interpolate_DiscardRow.log.diff 

missing_data/test_missing_data_air_passengers_Interpolate_Interpolate.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_air_passengers_Interpolate_Interpolate.py  >  logs/missing_data_test_missing_data_air_passengers_Interpolate_Interpolate.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_air_passengers_Interpolate_Interpolate.log logs/missing_data_test_missing_data_air_passengers_Interpolate_Interpolate.log  >  logs/missing_data_test_missing_data_air_passengers_Interpolate_Interpolate.log.diff
	 tail -10  logs/missing_data_test_missing_data_air_passengers_Interpolate_Interpolate.log.diff 

missing_data/test_missing_data_air_passengers_Interpolate_Mean.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_air_passengers_Interpolate_Mean.py  >  logs/missing_data_test_missing_data_air_passengers_Interpolate_Mean.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_air_passengers_Interpolate_Mean.log logs/missing_data_test_missing_data_air_passengers_Interpolate_Mean.log  >  logs/missing_data_test_missing_data_air_passengers_Interpolate_Mean.log.diff
	 tail -10  logs/missing_data_test_missing_data_air_passengers_Interpolate_Mean.log.diff 

missing_data/test_missing_data_air_passengers_Interpolate_Median.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_air_passengers_Interpolate_Median.py  >  logs/missing_data_test_missing_data_air_passengers_Interpolate_Median.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_air_passengers_Interpolate_Median.log logs/missing_data_test_missing_data_air_passengers_Interpolate_Median.log  >  logs/missing_data_test_missing_data_air_passengers_Interpolate_Median.log.diff
	 tail -10  logs/missing_data_test_missing_data_air_passengers_Interpolate_Median.log.diff 

missing_data/test_missing_data_air_passengers_Interpolate_None.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_air_passengers_Interpolate_None.py  >  logs/missing_data_test_missing_data_air_passengers_Interpolate_None.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_air_passengers_Interpolate_None.log logs/missing_data_test_missing_data_air_passengers_Interpolate_None.log  >  logs/missing_data_test_missing_data_air_passengers_Interpolate_None.log.diff
	 tail -10  logs/missing_data_test_missing_data_air_passengers_Interpolate_None.log.diff 

missing_data/test_missing_data_air_passengers_Interpolate_PreviousValue.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_air_passengers_Interpolate_PreviousValue.py  >  logs/missing_data_test_missing_data_air_passengers_Interpolate_PreviousValue.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_air_passengers_Interpolate_PreviousValue.log logs/missing_data_test_missing_data_air_passengers_Interpolate_PreviousValue.log  >  logs/missing_data_test_missing_data_air_passengers_Interpolate_PreviousValue.log.diff
	 tail -10  logs/missing_data_test_missing_data_air_passengers_Interpolate_PreviousValue.log.diff 

missing_data/test_missing_data_air_passengers_None_Constant.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_air_passengers_None_Constant.py  >  logs/missing_data_test_missing_data_air_passengers_None_Constant.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_air_passengers_None_Constant.log logs/missing_data_test_missing_data_air_passengers_None_Constant.log  >  logs/missing_data_test_missing_data_air_passengers_None_Constant.log.diff
	 tail -10  logs/missing_data_test_missing_data_air_passengers_None_Constant.log.diff 

missing_data/test_missing_data_air_passengers_None_DiscardRow.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_air_passengers_None_DiscardRow.py  >  logs/missing_data_test_missing_data_air_passengers_None_DiscardRow.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_air_passengers_None_DiscardRow.log logs/missing_data_test_missing_data_air_passengers_None_DiscardRow.log  >  logs/missing_data_test_missing_data_air_passengers_None_DiscardRow.log.diff
	 tail -10  logs/missing_data_test_missing_data_air_passengers_None_DiscardRow.log.diff 

missing_data/test_missing_data_air_passengers_None_Interpolate.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_air_passengers_None_Interpolate.py  >  logs/missing_data_test_missing_data_air_passengers_None_Interpolate.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_air_passengers_None_Interpolate.log logs/missing_data_test_missing_data_air_passengers_None_Interpolate.log  >  logs/missing_data_test_missing_data_air_passengers_None_Interpolate.log.diff
	 tail -10  logs/missing_data_test_missing_data_air_passengers_None_Interpolate.log.diff 

missing_data/test_missing_data_air_passengers_None_Mean.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_air_passengers_None_Mean.py  >  logs/missing_data_test_missing_data_air_passengers_None_Mean.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_air_passengers_None_Mean.log logs/missing_data_test_missing_data_air_passengers_None_Mean.log  >  logs/missing_data_test_missing_data_air_passengers_None_Mean.log.diff
	 tail -10  logs/missing_data_test_missing_data_air_passengers_None_Mean.log.diff 

missing_data/test_missing_data_air_passengers_None_Median.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_air_passengers_None_Median.py  >  logs/missing_data_test_missing_data_air_passengers_None_Median.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_air_passengers_None_Median.log logs/missing_data_test_missing_data_air_passengers_None_Median.log  >  logs/missing_data_test_missing_data_air_passengers_None_Median.log.diff
	 tail -10  logs/missing_data_test_missing_data_air_passengers_None_Median.log.diff 

missing_data/test_missing_data_air_passengers_None_None.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_air_passengers_None_None.py  >  logs/missing_data_test_missing_data_air_passengers_None_None.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_air_passengers_None_None.log logs/missing_data_test_missing_data_air_passengers_None_None.log  >  logs/missing_data_test_missing_data_air_passengers_None_None.log.diff
	 tail -10  logs/missing_data_test_missing_data_air_passengers_None_None.log.diff 

missing_data/test_missing_data_air_passengers_None_PreviousValue.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_air_passengers_None_PreviousValue.py  >  logs/missing_data_test_missing_data_air_passengers_None_PreviousValue.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_air_passengers_None_PreviousValue.log logs/missing_data_test_missing_data_air_passengers_None_PreviousValue.log  >  logs/missing_data_test_missing_data_air_passengers_None_PreviousValue.log.diff
	 tail -10  logs/missing_data_test_missing_data_air_passengers_None_PreviousValue.log.diff 

missing_data/test_missing_data_air_passengers_generic.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_air_passengers_generic.py  >  logs/missing_data_test_missing_data_air_passengers_generic.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_air_passengers_generic.log logs/missing_data_test_missing_data_air_passengers_generic.log  >  logs/missing_data_test_missing_data_air_passengers_generic.log.diff
	 tail -10  logs/missing_data_test_missing_data_air_passengers_generic.log.diff 

missing_data/test_missing_data_ozone_DiscardRow_Constant.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_ozone_DiscardRow_Constant.py  >  logs/missing_data_test_missing_data_ozone_DiscardRow_Constant.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_ozone_DiscardRow_Constant.log logs/missing_data_test_missing_data_ozone_DiscardRow_Constant.log  >  logs/missing_data_test_missing_data_ozone_DiscardRow_Constant.log.diff
	 tail -10  logs/missing_data_test_missing_data_ozone_DiscardRow_Constant.log.diff 

missing_data/test_missing_data_ozone_DiscardRow_DiscardRow.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_ozone_DiscardRow_DiscardRow.py  >  logs/missing_data_test_missing_data_ozone_DiscardRow_DiscardRow.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_ozone_DiscardRow_DiscardRow.log logs/missing_data_test_missing_data_ozone_DiscardRow_DiscardRow.log  >  logs/missing_data_test_missing_data_ozone_DiscardRow_DiscardRow.log.diff
	 tail -10  logs/missing_data_test_missing_data_ozone_DiscardRow_DiscardRow.log.diff 

missing_data/test_missing_data_ozone_DiscardRow_Interpolate.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_ozone_DiscardRow_Interpolate.py  >  logs/missing_data_test_missing_data_ozone_DiscardRow_Interpolate.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_ozone_DiscardRow_Interpolate.log logs/missing_data_test_missing_data_ozone_DiscardRow_Interpolate.log  >  logs/missing_data_test_missing_data_ozone_DiscardRow_Interpolate.log.diff
	 tail -10  logs/missing_data_test_missing_data_ozone_DiscardRow_Interpolate.log.diff 

missing_data/test_missing_data_ozone_DiscardRow_Mean.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_ozone_DiscardRow_Mean.py  >  logs/missing_data_test_missing_data_ozone_DiscardRow_Mean.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_ozone_DiscardRow_Mean.log logs/missing_data_test_missing_data_ozone_DiscardRow_Mean.log  >  logs/missing_data_test_missing_data_ozone_DiscardRow_Mean.log.diff
	 tail -10  logs/missing_data_test_missing_data_ozone_DiscardRow_Mean.log.diff 

missing_data/test_missing_data_ozone_DiscardRow_Median.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_ozone_DiscardRow_Median.py  >  logs/missing_data_test_missing_data_ozone_DiscardRow_Median.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_ozone_DiscardRow_Median.log logs/missing_data_test_missing_data_ozone_DiscardRow_Median.log  >  logs/missing_data_test_missing_data_ozone_DiscardRow_Median.log.diff
	 tail -10  logs/missing_data_test_missing_data_ozone_DiscardRow_Median.log.diff 

missing_data/test_missing_data_ozone_DiscardRow_None.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_ozone_DiscardRow_None.py  >  logs/missing_data_test_missing_data_ozone_DiscardRow_None.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_ozone_DiscardRow_None.log logs/missing_data_test_missing_data_ozone_DiscardRow_None.log  >  logs/missing_data_test_missing_data_ozone_DiscardRow_None.log.diff
	 tail -10  logs/missing_data_test_missing_data_ozone_DiscardRow_None.log.diff 

missing_data/test_missing_data_ozone_DiscardRow_PreviousValue.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_ozone_DiscardRow_PreviousValue.py  >  logs/missing_data_test_missing_data_ozone_DiscardRow_PreviousValue.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_ozone_DiscardRow_PreviousValue.log logs/missing_data_test_missing_data_ozone_DiscardRow_PreviousValue.log  >  logs/missing_data_test_missing_data_ozone_DiscardRow_PreviousValue.log.diff
	 tail -10  logs/missing_data_test_missing_data_ozone_DiscardRow_PreviousValue.log.diff 

missing_data/test_missing_data_ozone_Interpolate_Constant.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_ozone_Interpolate_Constant.py  >  logs/missing_data_test_missing_data_ozone_Interpolate_Constant.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_ozone_Interpolate_Constant.log logs/missing_data_test_missing_data_ozone_Interpolate_Constant.log  >  logs/missing_data_test_missing_data_ozone_Interpolate_Constant.log.diff
	 tail -10  logs/missing_data_test_missing_data_ozone_Interpolate_Constant.log.diff 

missing_data/test_missing_data_ozone_Interpolate_DiscardRow.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_ozone_Interpolate_DiscardRow.py  >  logs/missing_data_test_missing_data_ozone_Interpolate_DiscardRow.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_ozone_Interpolate_DiscardRow.log logs/missing_data_test_missing_data_ozone_Interpolate_DiscardRow.log  >  logs/missing_data_test_missing_data_ozone_Interpolate_DiscardRow.log.diff
	 tail -10  logs/missing_data_test_missing_data_ozone_Interpolate_DiscardRow.log.diff 

missing_data/test_missing_data_ozone_Interpolate_Interpolate.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_ozone_Interpolate_Interpolate.py  >  logs/missing_data_test_missing_data_ozone_Interpolate_Interpolate.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_ozone_Interpolate_Interpolate.log logs/missing_data_test_missing_data_ozone_Interpolate_Interpolate.log  >  logs/missing_data_test_missing_data_ozone_Interpolate_Interpolate.log.diff
	 tail -10  logs/missing_data_test_missing_data_ozone_Interpolate_Interpolate.log.diff 

missing_data/test_missing_data_ozone_Interpolate_Mean.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_ozone_Interpolate_Mean.py  >  logs/missing_data_test_missing_data_ozone_Interpolate_Mean.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_ozone_Interpolate_Mean.log logs/missing_data_test_missing_data_ozone_Interpolate_Mean.log  >  logs/missing_data_test_missing_data_ozone_Interpolate_Mean.log.diff
	 tail -10  logs/missing_data_test_missing_data_ozone_Interpolate_Mean.log.diff 

missing_data/test_missing_data_ozone_Interpolate_Median.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_ozone_Interpolate_Median.py  >  logs/missing_data_test_missing_data_ozone_Interpolate_Median.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_ozone_Interpolate_Median.log logs/missing_data_test_missing_data_ozone_Interpolate_Median.log  >  logs/missing_data_test_missing_data_ozone_Interpolate_Median.log.diff
	 tail -10  logs/missing_data_test_missing_data_ozone_Interpolate_Median.log.diff 

missing_data/test_missing_data_ozone_Interpolate_None.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_ozone_Interpolate_None.py  >  logs/missing_data_test_missing_data_ozone_Interpolate_None.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_ozone_Interpolate_None.log logs/missing_data_test_missing_data_ozone_Interpolate_None.log  >  logs/missing_data_test_missing_data_ozone_Interpolate_None.log.diff
	 tail -10  logs/missing_data_test_missing_data_ozone_Interpolate_None.log.diff 

missing_data/test_missing_data_ozone_Interpolate_PreviousValue.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_ozone_Interpolate_PreviousValue.py  >  logs/missing_data_test_missing_data_ozone_Interpolate_PreviousValue.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_ozone_Interpolate_PreviousValue.log logs/missing_data_test_missing_data_ozone_Interpolate_PreviousValue.log  >  logs/missing_data_test_missing_data_ozone_Interpolate_PreviousValue.log.diff
	 tail -10  logs/missing_data_test_missing_data_ozone_Interpolate_PreviousValue.log.diff 

missing_data/test_missing_data_ozone_None_Constant.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_ozone_None_Constant.py  >  logs/missing_data_test_missing_data_ozone_None_Constant.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_ozone_None_Constant.log logs/missing_data_test_missing_data_ozone_None_Constant.log  >  logs/missing_data_test_missing_data_ozone_None_Constant.log.diff
	 tail -10  logs/missing_data_test_missing_data_ozone_None_Constant.log.diff 

missing_data/test_missing_data_ozone_None_DiscardRow.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_ozone_None_DiscardRow.py  >  logs/missing_data_test_missing_data_ozone_None_DiscardRow.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_ozone_None_DiscardRow.log logs/missing_data_test_missing_data_ozone_None_DiscardRow.log  >  logs/missing_data_test_missing_data_ozone_None_DiscardRow.log.diff
	 tail -10  logs/missing_data_test_missing_data_ozone_None_DiscardRow.log.diff 

missing_data/test_missing_data_ozone_None_Interpolate.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_ozone_None_Interpolate.py  >  logs/missing_data_test_missing_data_ozone_None_Interpolate.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_ozone_None_Interpolate.log logs/missing_data_test_missing_data_ozone_None_Interpolate.log  >  logs/missing_data_test_missing_data_ozone_None_Interpolate.log.diff
	 tail -10  logs/missing_data_test_missing_data_ozone_None_Interpolate.log.diff 

missing_data/test_missing_data_ozone_None_Mean.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_ozone_None_Mean.py  >  logs/missing_data_test_missing_data_ozone_None_Mean.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_ozone_None_Mean.log logs/missing_data_test_missing_data_ozone_None_Mean.log  >  logs/missing_data_test_missing_data_ozone_None_Mean.log.diff
	 tail -10  logs/missing_data_test_missing_data_ozone_None_Mean.log.diff 

missing_data/test_missing_data_ozone_None_Median.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_ozone_None_Median.py  >  logs/missing_data_test_missing_data_ozone_None_Median.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_ozone_None_Median.log logs/missing_data_test_missing_data_ozone_None_Median.log  >  logs/missing_data_test_missing_data_ozone_None_Median.log.diff
	 tail -10  logs/missing_data_test_missing_data_ozone_None_Median.log.diff 

missing_data/test_missing_data_ozone_None_None.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_ozone_None_None.py  >  logs/missing_data_test_missing_data_ozone_None_None.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_ozone_None_None.log logs/missing_data_test_missing_data_ozone_None_None.log  >  logs/missing_data_test_missing_data_ozone_None_None.log.diff
	 tail -10  logs/missing_data_test_missing_data_ozone_None_None.log.diff 

missing_data/test_missing_data_ozone_None_PreviousValue.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_ozone_None_PreviousValue.py  >  logs/missing_data_test_missing_data_ozone_None_PreviousValue.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_ozone_None_PreviousValue.log logs/missing_data_test_missing_data_ozone_None_PreviousValue.log  >  logs/missing_data_test_missing_data_ozone_None_PreviousValue.log.diff
	 tail -10  logs/missing_data_test_missing_data_ozone_None_PreviousValue.log.diff 

missing_data/test_missing_data_ozone_generic.py  :  
	 -$(PYTHON)  tests/missing_data/test_missing_data_ozone_generic.py  >  logs/missing_data_test_missing_data_ozone_generic.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/missing_data/test_missing_data_ozone_generic.log logs/missing_data_test_missing_data_ozone_generic.log  >  logs/missing_data_test_missing_data_ozone_generic.log.diff
	 tail -10  logs/missing_data_test_missing_data_ozone_generic.log.diff 



 missing_data :  missing_data/test_missing_data_ozone_generic.py missing_data/test_missing_data_ozone_None_PreviousValue.py missing_data/test_missing_data_ozone_None_None.py missing_data/test_missing_data_ozone_None_Median.py missing_data/test_missing_data_ozone_None_Mean.py missing_data/test_missing_data_ozone_None_Interpolate.py missing_data/test_missing_data_ozone_None_DiscardRow.py missing_data/test_missing_data_ozone_None_Constant.py missing_data/test_missing_data_ozone_Interpolate_PreviousValue.py missing_data/test_missing_data_ozone_Interpolate_None.py missing_data/test_missing_data_ozone_Interpolate_Median.py missing_data/test_missing_data_ozone_Interpolate_Mean.py missing_data/test_missing_data_ozone_Interpolate_Interpolate.py missing_data/test_missing_data_ozone_Interpolate_DiscardRow.py missing_data/test_missing_data_ozone_Interpolate_Constant.py missing_data/test_missing_data_ozone_DiscardRow_PreviousValue.py missing_data/test_missing_data_ozone_DiscardRow_None.py missing_data/test_missing_data_ozone_DiscardRow_Median.py missing_data/test_missing_data_ozone_DiscardRow_Mean.py missing_data/test_missing_data_ozone_DiscardRow_Interpolate.py missing_data/test_missing_data_ozone_DiscardRow_DiscardRow.py missing_data/test_missing_data_ozone_DiscardRow_Constant.py missing_data/test_missing_data_air_passengers_generic.py missing_data/test_missing_data_air_passengers_None_PreviousValue.py missing_data/test_missing_data_air_passengers_None_None.py missing_data/test_missing_data_air_passengers_None_Median.py missing_data/test_missing_data_air_passengers_None_Mean.py missing_data/test_missing_data_air_passengers_None_Interpolate.py missing_data/test_missing_data_air_passengers_None_DiscardRow.py missing_data/test_missing_data_air_passengers_None_Constant.py missing_data/test_missing_data_air_passengers_Interpolate_PreviousValue.py missing_data/test_missing_data_air_passengers_Interpolate_None.py missing_data/test_missing_data_air_passengers_Interpolate_Median.py missing_data/test_missing_data_air_passengers_Interpolate_Mean.py missing_data/test_missing_data_air_passengers_Interpolate_Interpolate.py missing_data/test_missing_data_air_passengers_Interpolate_DiscardRow.py missing_data/test_missing_data_air_passengers_Interpolate_Constant.py missing_data/test_missing_data_air_passengers_DiscardRow_PreviousValue.py missing_data/test_missing_data_air_passengers_DiscardRow_None.py missing_data/test_missing_data_air_passengers_DiscardRow_Median.py missing_data/test_missing_data_air_passengers_DiscardRow_Mean.py missing_data/test_missing_data_air_passengers_DiscardRow_Interpolate.py missing_data/test_missing_data_air_passengers_DiscardRow_DiscardRow.py missing_data/test_missing_data_air_passengers_DiscardRow_Constant.py missing_data/gen_ozone_tests.py missing_data/gen_air_passengers_tests.py  
 

model_control/test_ozone_all_models_enabled.py  :  
	 -$(PYTHON)  tests/model_control/test_ozone_all_models_enabled.py  >  logs/model_control_test_ozone_all_models_enabled.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/model_control/test_ozone_all_models_enabled.log logs/model_control_test_ozone_all_models_enabled.log  >  logs/model_control_test_ozone_all_models_enabled.log.diff
	 tail -10  logs/model_control_test_ozone_all_models_enabled.log.diff 

model_control/test_ozone_custom_models_enabled.py  :  
	 -$(PYTHON)  tests/model_control/test_ozone_custom_models_enabled.py  >  logs/model_control_test_ozone_custom_models_enabled.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/model_control/test_ozone_custom_models_enabled.log logs/model_control_test_ozone_custom_models_enabled.log  >  logs/model_control_test_ozone_custom_models_enabled.log.diff
	 tail -10  logs/model_control_test_ozone_custom_models_enabled.log.diff 

model_control/test_ozone_no_models_enabled.py  :  
	 -$(PYTHON)  tests/model_control/test_ozone_no_models_enabled.py  >  logs/model_control_test_ozone_no_models_enabled.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/model_control/test_ozone_no_models_enabled.log logs/model_control_test_ozone_no_models_enabled.log  >  logs/model_control_test_ozone_no_models_enabled.log.diff
	 tail -10  logs/model_control_test_ozone_no_models_enabled.log.diff 



 model_control :  model_control/test_ozone_no_models_enabled.py model_control/test_ozone_custom_models_enabled.py model_control/test_ozone_all_models_enabled.py  
 

multiplicative_seasonal/artificial_model_Constant.py  :  
	 -$(PYTHON)  tests/multiplicative_seasonal/artificial_model_Constant.py  >  logs/multiplicative_seasonal_artificial_model_Constant.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/multiplicative_seasonal/artificial_model_Constant.log logs/multiplicative_seasonal_artificial_model_Constant.log  >  logs/multiplicative_seasonal_artificial_model_Constant.log.diff
	 tail -10  logs/multiplicative_seasonal_artificial_model_Constant.log.diff 

multiplicative_seasonal/artificial_model_One.py  :  
	 -$(PYTHON)  tests/multiplicative_seasonal/artificial_model_One.py  >  logs/multiplicative_seasonal_artificial_model_One.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/multiplicative_seasonal/artificial_model_One.log logs/multiplicative_seasonal_artificial_model_One.log  >  logs/multiplicative_seasonal_artificial_model_One.log.diff
	 tail -10  logs/multiplicative_seasonal_artificial_model_One.log.diff 

multiplicative_seasonal/artificial_model_T+S+R.py  :  
	 -$(PYTHON)  tests/multiplicative_seasonal/artificial_model_T+S+R.py  >  logs/multiplicative_seasonal_artificial_model_T+S+R.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/multiplicative_seasonal/artificial_model_T+S+R.log logs/multiplicative_seasonal_artificial_model_T+S+R.log  >  logs/multiplicative_seasonal_artificial_model_T+S+R.log.diff
	 tail -10  logs/multiplicative_seasonal_artificial_model_T+S+R.log.diff 

multiplicative_seasonal/artificial_model_TS+R.py  :  
	 -$(PYTHON)  tests/multiplicative_seasonal/artificial_model_TS+R.py  >  logs/multiplicative_seasonal_artificial_model_TS+R.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/multiplicative_seasonal/artificial_model_TS+R.log logs/multiplicative_seasonal_artificial_model_TS+R.log  >  logs/multiplicative_seasonal_artificial_model_TS+R.log.diff
	 tail -10  logs/multiplicative_seasonal_artificial_model_TS+R.log.diff 

multiplicative_seasonal/artificial_model_TSR.py  :  
	 -$(PYTHON)  tests/multiplicative_seasonal/artificial_model_TSR.py  >  logs/multiplicative_seasonal_artificial_model_TSR.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/multiplicative_seasonal/artificial_model_TSR.log logs/multiplicative_seasonal_artificial_model_TSR.log  >  logs/multiplicative_seasonal_artificial_model_TSR.log.diff
	 tail -10  logs/multiplicative_seasonal_artificial_model_TSR.log.diff 

multiplicative_seasonal/artificial_model_Zero.py  :  
	 -$(PYTHON)  tests/multiplicative_seasonal/artificial_model_Zero.py  >  logs/multiplicative_seasonal_artificial_model_Zero.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/multiplicative_seasonal/artificial_model_Zero.log logs/multiplicative_seasonal_artificial_model_Zero.log  >  logs/multiplicative_seasonal_artificial_model_Zero.log.diff
	 tail -10  logs/multiplicative_seasonal_artificial_model_Zero.log.diff 

multiplicative_seasonal/test_air_passengers_multiplicative_seasonal_T+S+R.py  :  
	 -$(PYTHON)  tests/multiplicative_seasonal/test_air_passengers_multiplicative_seasonal_T+S+R.py  >  logs/multiplicative_seasonal_test_air_passengers_multiplicative_seasonal_T+S+R.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/multiplicative_seasonal/test_air_passengers_multiplicative_seasonal_T+S+R.log logs/multiplicative_seasonal_test_air_passengers_multiplicative_seasonal_T+S+R.log  >  logs/multiplicative_seasonal_test_air_passengers_multiplicative_seasonal_T+S+R.log.diff
	 tail -10  logs/multiplicative_seasonal_test_air_passengers_multiplicative_seasonal_T+S+R.log.diff 

multiplicative_seasonal/test_air_passengers_multiplicative_seasonal_TS+R.py  :  
	 -$(PYTHON)  tests/multiplicative_seasonal/test_air_passengers_multiplicative_seasonal_TS+R.py  >  logs/multiplicative_seasonal_test_air_passengers_multiplicative_seasonal_TS+R.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/multiplicative_seasonal/test_air_passengers_multiplicative_seasonal_TS+R.log logs/multiplicative_seasonal_test_air_passengers_multiplicative_seasonal_TS+R.log  >  logs/multiplicative_seasonal_test_air_passengers_multiplicative_seasonal_TS+R.log.diff
	 tail -10  logs/multiplicative_seasonal_test_air_passengers_multiplicative_seasonal_TS+R.log.diff 

multiplicative_seasonal/test_air_passengers_multiplicative_seasonal_TSR.py  :  
	 -$(PYTHON)  tests/multiplicative_seasonal/test_air_passengers_multiplicative_seasonal_TSR.py  >  logs/multiplicative_seasonal_test_air_passengers_multiplicative_seasonal_TSR.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/multiplicative_seasonal/test_air_passengers_multiplicative_seasonal_TSR.log logs/multiplicative_seasonal_test_air_passengers_multiplicative_seasonal_TSR.log  >  logs/multiplicative_seasonal_test_air_passengers_multiplicative_seasonal_TSR.log.diff
	 tail -10  logs/multiplicative_seasonal_test_air_passengers_multiplicative_seasonal_TSR.log.diff 

multiplicative_seasonal/test_air_passengers_multiplicative_seasonal_ssslow_mode.py  :  
	 -$(PYTHON)  tests/multiplicative_seasonal/test_air_passengers_multiplicative_seasonal_ssslow_mode.py  >  logs/multiplicative_seasonal_test_air_passengers_multiplicative_seasonal_ssslow_mode.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/multiplicative_seasonal/test_air_passengers_multiplicative_seasonal_ssslow_mode.log logs/multiplicative_seasonal_test_air_passengers_multiplicative_seasonal_ssslow_mode.log  >  logs/multiplicative_seasonal_test_air_passengers_multiplicative_seasonal_ssslow_mode.log.diff
	 tail -10  logs/multiplicative_seasonal_test_air_passengers_multiplicative_seasonal_ssslow_mode.log.diff 

multiplicative_seasonal/test_ozone_multiplicative_seasonal_T+S+R.py  :  
	 -$(PYTHON)  tests/multiplicative_seasonal/test_ozone_multiplicative_seasonal_T+S+R.py  >  logs/multiplicative_seasonal_test_ozone_multiplicative_seasonal_T+S+R.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/multiplicative_seasonal/test_ozone_multiplicative_seasonal_T+S+R.log logs/multiplicative_seasonal_test_ozone_multiplicative_seasonal_T+S+R.log  >  logs/multiplicative_seasonal_test_ozone_multiplicative_seasonal_T+S+R.log.diff
	 tail -10  logs/multiplicative_seasonal_test_ozone_multiplicative_seasonal_T+S+R.log.diff 

multiplicative_seasonal/test_ozone_multiplicative_seasonal_TS+R.py  :  
	 -$(PYTHON)  tests/multiplicative_seasonal/test_ozone_multiplicative_seasonal_TS+R.py  >  logs/multiplicative_seasonal_test_ozone_multiplicative_seasonal_TS+R.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/multiplicative_seasonal/test_ozone_multiplicative_seasonal_TS+R.log logs/multiplicative_seasonal_test_ozone_multiplicative_seasonal_TS+R.log  >  logs/multiplicative_seasonal_test_ozone_multiplicative_seasonal_TS+R.log.diff
	 tail -10  logs/multiplicative_seasonal_test_ozone_multiplicative_seasonal_TS+R.log.diff 

multiplicative_seasonal/test_ozone_multiplicative_seasonal_TSR.py  :  
	 -$(PYTHON)  tests/multiplicative_seasonal/test_ozone_multiplicative_seasonal_TSR.py  >  logs/multiplicative_seasonal_test_ozone_multiplicative_seasonal_TSR.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/multiplicative_seasonal/test_ozone_multiplicative_seasonal_TSR.log logs/multiplicative_seasonal_test_ozone_multiplicative_seasonal_TSR.log  >  logs/multiplicative_seasonal_test_ozone_multiplicative_seasonal_TSR.log.diff
	 tail -10  logs/multiplicative_seasonal_test_ozone_multiplicative_seasonal_TSR.log.diff 

multiplicative_seasonal/test_ozone_multiplicative_seasonal_ssslow_mode.py  :  
	 -$(PYTHON)  tests/multiplicative_seasonal/test_ozone_multiplicative_seasonal_ssslow_mode.py  >  logs/multiplicative_seasonal_test_ozone_multiplicative_seasonal_ssslow_mode.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/multiplicative_seasonal/test_ozone_multiplicative_seasonal_ssslow_mode.log logs/multiplicative_seasonal_test_ozone_multiplicative_seasonal_ssslow_mode.log  >  logs/multiplicative_seasonal_test_ozone_multiplicative_seasonal_ssslow_mode.log.diff
	 tail -10  logs/multiplicative_seasonal_test_ozone_multiplicative_seasonal_ssslow_mode.log.diff 



 multiplicative_seasonal :  multiplicative_seasonal/test_ozone_multiplicative_seasonal_ssslow_mode.py multiplicative_seasonal/test_ozone_multiplicative_seasonal_TSR.py multiplicative_seasonal/test_ozone_multiplicative_seasonal_TS+R.py multiplicative_seasonal/test_ozone_multiplicative_seasonal_T+S+R.py multiplicative_seasonal/test_air_passengers_multiplicative_seasonal_ssslow_mode.py multiplicative_seasonal/test_air_passengers_multiplicative_seasonal_TSR.py multiplicative_seasonal/test_air_passengers_multiplicative_seasonal_TS+R.py multiplicative_seasonal/test_air_passengers_multiplicative_seasonal_T+S+R.py multiplicative_seasonal/artificial_model_Zero.py multiplicative_seasonal/artificial_model_TSR.py multiplicative_seasonal/artificial_model_TS+R.py multiplicative_seasonal/artificial_model_T+S+R.py multiplicative_seasonal/artificial_model_One.py multiplicative_seasonal/artificial_model_Constant.py  
 

neuralnet/test_air_passengers_CPU_theano.py  :  
	 -$(PYTHON)  tests/neuralnet/test_air_passengers_CPU_theano.py  >  logs/neuralnet_test_air_passengers_CPU_theano.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/neuralnet/test_air_passengers_CPU_theano.log logs/neuralnet_test_air_passengers_CPU_theano.log  >  logs/neuralnet_test_air_passengers_CPU_theano.log.diff
	 tail -10  logs/neuralnet_test_air_passengers_CPU_theano.log.diff 

neuralnet/test_air_passengers_GPU.py  :  
	 -$(PYTHON)  tests/neuralnet/test_air_passengers_GPU.py  >  logs/neuralnet_test_air_passengers_GPU.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/neuralnet/test_air_passengers_GPU.log logs/neuralnet_test_air_passengers_GPU.log  >  logs/neuralnet_test_air_passengers_GPU.log.diff
	 tail -10  logs/neuralnet_test_air_passengers_GPU.log.diff 

neuralnet/test_air_passengers_GPU_tensorflow.py  :  
	 -$(PYTHON)  tests/neuralnet/test_air_passengers_GPU_tensorflow.py  >  logs/neuralnet_test_air_passengers_GPU_tensorflow.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/neuralnet/test_air_passengers_GPU_tensorflow.log logs/neuralnet_test_air_passengers_GPU_tensorflow.log  >  logs/neuralnet_test_air_passengers_GPU_tensorflow.log.diff
	 tail -10  logs/neuralnet_test_air_passengers_GPU_tensorflow.log.diff 

neuralnet/test_air_passengers_GPU_theano.py  :  
	 -$(PYTHON)  tests/neuralnet/test_air_passengers_GPU_theano.py  >  logs/neuralnet_test_air_passengers_GPU_theano.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/neuralnet/test_air_passengers_GPU_theano.log logs/neuralnet_test_air_passengers_GPU_theano.log  >  logs/neuralnet_test_air_passengers_GPU_theano.log.diff
	 tail -10  logs/neuralnet_test_air_passengers_GPU_theano.log.diff 

neuralnet/test_air_passengers_rnn_only.py  :  
	 -$(PYTHON)  tests/neuralnet/test_air_passengers_rnn_only.py  >  logs/neuralnet_test_air_passengers_rnn_only.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/neuralnet/test_air_passengers_rnn_only.log logs/neuralnet_test_air_passengers_rnn_only.log  >  logs/neuralnet_test_air_passengers_rnn_only.log.diff
	 tail -10  logs/neuralnet_test_air_passengers_rnn_only.log.diff 

neuralnet/test_air_passengers_tensorflow.py  :  
	 -$(PYTHON)  tests/neuralnet/test_air_passengers_tensorflow.py  >  logs/neuralnet_test_air_passengers_tensorflow.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/neuralnet/test_air_passengers_tensorflow.log logs/neuralnet_test_air_passengers_tensorflow.log  >  logs/neuralnet_test_air_passengers_tensorflow.log.diff
	 tail -10  logs/neuralnet_test_air_passengers_tensorflow.log.diff 

neuralnet/test_ozone_GPU.py  :  
	 -$(PYTHON)  tests/neuralnet/test_ozone_GPU.py  >  logs/neuralnet_test_ozone_GPU.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/neuralnet/test_ozone_GPU.log logs/neuralnet_test_ozone_GPU.log  >  logs/neuralnet_test_ozone_GPU.log.diff
	 tail -10  logs/neuralnet_test_ozone_GPU.log.diff 

neuralnet/test_ozone__CPU_theano.py  :  
	 -$(PYTHON)  tests/neuralnet/test_ozone__CPU_theano.py  >  logs/neuralnet_test_ozone__CPU_theano.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/neuralnet/test_ozone__CPU_theano.log logs/neuralnet_test_ozone__CPU_theano.log  >  logs/neuralnet_test_ozone__CPU_theano.log.diff
	 tail -10  logs/neuralnet_test_ozone__CPU_theano.log.diff 

neuralnet/test_ozone__GPU_tensorflow.py  :  
	 -$(PYTHON)  tests/neuralnet/test_ozone__GPU_tensorflow.py  >  logs/neuralnet_test_ozone__GPU_tensorflow.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/neuralnet/test_ozone__GPU_tensorflow.log logs/neuralnet_test_ozone__GPU_tensorflow.log  >  logs/neuralnet_test_ozone__GPU_tensorflow.log.diff
	 tail -10  logs/neuralnet_test_ozone__GPU_tensorflow.log.diff 

neuralnet/test_ozone__GPU_theano.py  :  
	 -$(PYTHON)  tests/neuralnet/test_ozone__GPU_theano.py  >  logs/neuralnet_test_ozone__GPU_theano.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/neuralnet/test_ozone__GPU_theano.log logs/neuralnet_test_ozone__GPU_theano.log  >  logs/neuralnet_test_ozone__GPU_theano.log.diff
	 tail -10  logs/neuralnet_test_ozone__GPU_theano.log.diff 

neuralnet/test_ozone_rnn_only.py  :  
	 -$(PYTHON)  tests/neuralnet/test_ozone_rnn_only.py  >  logs/neuralnet_test_ozone_rnn_only.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/neuralnet/test_ozone_rnn_only.log logs/neuralnet_test_ozone_rnn_only.log  >  logs/neuralnet_test_ozone_rnn_only.log.diff
	 tail -10  logs/neuralnet_test_ozone_rnn_only.log.diff 

neuralnet/test_ozone_rnn_only_LSTM.py  :  
	 -$(PYTHON)  tests/neuralnet/test_ozone_rnn_only_LSTM.py  >  logs/neuralnet_test_ozone_rnn_only_LSTM.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/neuralnet/test_ozone_rnn_only_LSTM.log logs/neuralnet_test_ozone_rnn_only_LSTM.log  >  logs/neuralnet_test_ozone_rnn_only_LSTM.log.diff
	 tail -10  logs/neuralnet_test_ozone_rnn_only_LSTM.log.diff 

neuralnet/test_ozone_rnn_only_MLP.py  :  
	 -$(PYTHON)  tests/neuralnet/test_ozone_rnn_only_MLP.py  >  logs/neuralnet_test_ozone_rnn_only_MLP.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/neuralnet/test_ozone_rnn_only_MLP.log logs/neuralnet_test_ozone_rnn_only_MLP.log  >  logs/neuralnet_test_ozone_rnn_only_MLP.log.diff
	 tail -10  logs/neuralnet_test_ozone_rnn_only_MLP.log.diff 

neuralnet/test_ozone_tensorflow.py  :  
	 -$(PYTHON)  tests/neuralnet/test_ozone_tensorflow.py  >  logs/neuralnet_test_ozone_tensorflow.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/neuralnet/test_ozone_tensorflow.log logs/neuralnet_test_ozone_tensorflow.log  >  logs/neuralnet_test_ozone_tensorflow.log.diff
	 tail -10  logs/neuralnet_test_ozone_tensorflow.log.diff 



 neuralnet :  neuralnet/test_ozone_tensorflow.py neuralnet/test_ozone_rnn_only_MLP.py neuralnet/test_ozone_rnn_only_LSTM.py neuralnet/test_ozone_rnn_only.py neuralnet/test_ozone__GPU_theano.py neuralnet/test_ozone__GPU_tensorflow.py neuralnet/test_ozone__CPU_theano.py neuralnet/test_ozone_GPU.py neuralnet/test_air_passengers_tensorflow.py neuralnet/test_air_passengers_rnn_only.py neuralnet/test_air_passengers_GPU_theano.py neuralnet/test_air_passengers_GPU_tensorflow.py neuralnet/test_air_passengers_GPU.py neuralnet/test_air_passengers_CPU_theano.py  
 

perf/gen_ar_speed_tests.py  :  
	 -$(PYTHON)  tests/perf/gen_ar_speed_tests.py  >  logs/perf_gen_ar_speed_tests.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/gen_ar_speed_tests.log logs/perf_gen_ar_speed_tests.log  >  logs/perf_gen_ar_speed_tests.log.diff
	 tail -10  logs/perf_gen_ar_speed_tests.log.diff 

perf/gen_long_cycles_tests.py  :  
	 -$(PYTHON)  tests/perf/gen_long_cycles_tests.py  >  logs/perf_gen_long_cycles_tests.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/gen_long_cycles_tests.log logs/perf_gen_long_cycles_tests.log  >  logs/perf_gen_long_cycles_tests.log.diff
	 tail -10  logs/perf_gen_long_cycles_tests.log.diff 

perf/test_cycles_full_long.py  :  
	 -$(PYTHON)  tests/perf/test_cycles_full_long.py  >  logs/perf_test_cycles_full_long.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_cycles_full_long.log logs/perf_test_cycles_full_long.log  >  logs/perf_test_cycles_full_long.log.diff
	 tail -10  logs/perf_test_cycles_full_long.log.diff 

perf/test_cycles_full_long_long.py  :  
	 -$(PYTHON)  tests/perf/test_cycles_full_long_long.py  >  logs/perf_test_cycles_full_long_long.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_cycles_full_long_long.log logs/perf_test_cycles_full_long_long.log  >  logs/perf_test_cycles_full_long_long.log.diff
	 tail -10  logs/perf_test_cycles_full_long_long.log.diff 

perf/test_long_cycles_nbrows_cycle_length_1000_140.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_1000_140.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_1000_140.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_1000_140.log logs/perf_test_long_cycles_nbrows_cycle_length_1000_140.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_1000_140.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_1000_140.log.diff 

perf/test_long_cycles_nbrows_cycle_length_1000_20.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_1000_20.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_1000_20.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_1000_20.log logs/perf_test_long_cycles_nbrows_cycle_length_1000_20.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_1000_20.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_1000_20.log.diff 

perf/test_long_cycles_nbrows_cycle_length_1000_200.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_1000_200.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_1000_200.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_1000_200.log logs/perf_test_long_cycles_nbrows_cycle_length_1000_200.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_1000_200.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_1000_200.log.diff 

perf/test_long_cycles_nbrows_cycle_length_1000_260.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_1000_260.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_1000_260.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_1000_260.log logs/perf_test_long_cycles_nbrows_cycle_length_1000_260.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_1000_260.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_1000_260.log.diff 

perf/test_long_cycles_nbrows_cycle_length_1000_320.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_1000_320.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_1000_320.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_1000_320.log logs/perf_test_long_cycles_nbrows_cycle_length_1000_320.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_1000_320.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_1000_320.log.diff 

perf/test_long_cycles_nbrows_cycle_length_1000_380.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_1000_380.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_1000_380.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_1000_380.log logs/perf_test_long_cycles_nbrows_cycle_length_1000_380.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_1000_380.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_1000_380.log.diff 

perf/test_long_cycles_nbrows_cycle_length_1000_440.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_1000_440.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_1000_440.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_1000_440.log logs/perf_test_long_cycles_nbrows_cycle_length_1000_440.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_1000_440.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_1000_440.log.diff 

perf/test_long_cycles_nbrows_cycle_length_1000_80.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_1000_80.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_1000_80.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_1000_80.log logs/perf_test_long_cycles_nbrows_cycle_length_1000_80.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_1000_80.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_1000_80.log.diff 

perf/test_long_cycles_nbrows_cycle_length_11000_140.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_11000_140.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_11000_140.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_11000_140.log logs/perf_test_long_cycles_nbrows_cycle_length_11000_140.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_11000_140.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_11000_140.log.diff 

perf/test_long_cycles_nbrows_cycle_length_11000_20.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_11000_20.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_11000_20.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_11000_20.log logs/perf_test_long_cycles_nbrows_cycle_length_11000_20.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_11000_20.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_11000_20.log.diff 

perf/test_long_cycles_nbrows_cycle_length_11000_200.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_11000_200.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_11000_200.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_11000_200.log logs/perf_test_long_cycles_nbrows_cycle_length_11000_200.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_11000_200.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_11000_200.log.diff 

perf/test_long_cycles_nbrows_cycle_length_11000_260.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_11000_260.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_11000_260.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_11000_260.log logs/perf_test_long_cycles_nbrows_cycle_length_11000_260.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_11000_260.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_11000_260.log.diff 

perf/test_long_cycles_nbrows_cycle_length_11000_320.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_11000_320.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_11000_320.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_11000_320.log logs/perf_test_long_cycles_nbrows_cycle_length_11000_320.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_11000_320.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_11000_320.log.diff 

perf/test_long_cycles_nbrows_cycle_length_11000_380.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_11000_380.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_11000_380.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_11000_380.log logs/perf_test_long_cycles_nbrows_cycle_length_11000_380.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_11000_380.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_11000_380.log.diff 

perf/test_long_cycles_nbrows_cycle_length_11000_440.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_11000_440.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_11000_440.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_11000_440.log logs/perf_test_long_cycles_nbrows_cycle_length_11000_440.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_11000_440.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_11000_440.log.diff 

perf/test_long_cycles_nbrows_cycle_length_11000_80.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_11000_80.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_11000_80.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_11000_80.log logs/perf_test_long_cycles_nbrows_cycle_length_11000_80.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_11000_80.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_11000_80.log.diff 

perf/test_long_cycles_nbrows_cycle_length_21000_140.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_21000_140.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_21000_140.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_21000_140.log logs/perf_test_long_cycles_nbrows_cycle_length_21000_140.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_21000_140.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_21000_140.log.diff 

perf/test_long_cycles_nbrows_cycle_length_21000_20.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_21000_20.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_21000_20.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_21000_20.log logs/perf_test_long_cycles_nbrows_cycle_length_21000_20.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_21000_20.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_21000_20.log.diff 

perf/test_long_cycles_nbrows_cycle_length_21000_200.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_21000_200.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_21000_200.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_21000_200.log logs/perf_test_long_cycles_nbrows_cycle_length_21000_200.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_21000_200.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_21000_200.log.diff 

perf/test_long_cycles_nbrows_cycle_length_21000_260.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_21000_260.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_21000_260.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_21000_260.log logs/perf_test_long_cycles_nbrows_cycle_length_21000_260.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_21000_260.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_21000_260.log.diff 

perf/test_long_cycles_nbrows_cycle_length_21000_320.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_21000_320.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_21000_320.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_21000_320.log logs/perf_test_long_cycles_nbrows_cycle_length_21000_320.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_21000_320.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_21000_320.log.diff 

perf/test_long_cycles_nbrows_cycle_length_21000_380.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_21000_380.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_21000_380.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_21000_380.log logs/perf_test_long_cycles_nbrows_cycle_length_21000_380.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_21000_380.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_21000_380.log.diff 

perf/test_long_cycles_nbrows_cycle_length_21000_440.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_21000_440.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_21000_440.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_21000_440.log logs/perf_test_long_cycles_nbrows_cycle_length_21000_440.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_21000_440.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_21000_440.log.diff 

perf/test_long_cycles_nbrows_cycle_length_21000_80.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_21000_80.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_21000_80.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_21000_80.log logs/perf_test_long_cycles_nbrows_cycle_length_21000_80.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_21000_80.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_21000_80.log.diff 

perf/test_long_cycles_nbrows_cycle_length_31000_140.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_31000_140.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_31000_140.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_31000_140.log logs/perf_test_long_cycles_nbrows_cycle_length_31000_140.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_31000_140.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_31000_140.log.diff 

perf/test_long_cycles_nbrows_cycle_length_31000_20.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_31000_20.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_31000_20.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_31000_20.log logs/perf_test_long_cycles_nbrows_cycle_length_31000_20.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_31000_20.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_31000_20.log.diff 

perf/test_long_cycles_nbrows_cycle_length_31000_200.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_31000_200.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_31000_200.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_31000_200.log logs/perf_test_long_cycles_nbrows_cycle_length_31000_200.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_31000_200.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_31000_200.log.diff 

perf/test_long_cycles_nbrows_cycle_length_31000_260.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_31000_260.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_31000_260.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_31000_260.log logs/perf_test_long_cycles_nbrows_cycle_length_31000_260.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_31000_260.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_31000_260.log.diff 

perf/test_long_cycles_nbrows_cycle_length_31000_320.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_31000_320.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_31000_320.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_31000_320.log logs/perf_test_long_cycles_nbrows_cycle_length_31000_320.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_31000_320.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_31000_320.log.diff 

perf/test_long_cycles_nbrows_cycle_length_31000_380.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_31000_380.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_31000_380.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_31000_380.log logs/perf_test_long_cycles_nbrows_cycle_length_31000_380.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_31000_380.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_31000_380.log.diff 

perf/test_long_cycles_nbrows_cycle_length_31000_440.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_31000_440.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_31000_440.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_31000_440.log logs/perf_test_long_cycles_nbrows_cycle_length_31000_440.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_31000_440.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_31000_440.log.diff 

perf/test_long_cycles_nbrows_cycle_length_31000_80.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_31000_80.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_31000_80.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_31000_80.log logs/perf_test_long_cycles_nbrows_cycle_length_31000_80.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_31000_80.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_31000_80.log.diff 

perf/test_long_cycles_nbrows_cycle_length_41000_140.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_41000_140.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_41000_140.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_41000_140.log logs/perf_test_long_cycles_nbrows_cycle_length_41000_140.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_41000_140.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_41000_140.log.diff 

perf/test_long_cycles_nbrows_cycle_length_41000_20.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_41000_20.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_41000_20.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_41000_20.log logs/perf_test_long_cycles_nbrows_cycle_length_41000_20.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_41000_20.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_41000_20.log.diff 

perf/test_long_cycles_nbrows_cycle_length_41000_200.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_41000_200.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_41000_200.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_41000_200.log logs/perf_test_long_cycles_nbrows_cycle_length_41000_200.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_41000_200.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_41000_200.log.diff 

perf/test_long_cycles_nbrows_cycle_length_41000_260.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_41000_260.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_41000_260.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_41000_260.log logs/perf_test_long_cycles_nbrows_cycle_length_41000_260.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_41000_260.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_41000_260.log.diff 

perf/test_long_cycles_nbrows_cycle_length_41000_320.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_41000_320.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_41000_320.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_41000_320.log logs/perf_test_long_cycles_nbrows_cycle_length_41000_320.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_41000_320.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_41000_320.log.diff 

perf/test_long_cycles_nbrows_cycle_length_41000_380.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_41000_380.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_41000_380.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_41000_380.log logs/perf_test_long_cycles_nbrows_cycle_length_41000_380.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_41000_380.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_41000_380.log.diff 

perf/test_long_cycles_nbrows_cycle_length_41000_440.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_41000_440.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_41000_440.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_41000_440.log logs/perf_test_long_cycles_nbrows_cycle_length_41000_440.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_41000_440.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_41000_440.log.diff 

perf/test_long_cycles_nbrows_cycle_length_41000_80.py  :  
	 -$(PYTHON)  tests/perf/test_long_cycles_nbrows_cycle_length_41000_80.py  >  logs/perf_test_long_cycles_nbrows_cycle_length_41000_80.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_long_cycles_nbrows_cycle_length_41000_80.log logs/perf_test_long_cycles_nbrows_cycle_length_41000_80.log  >  logs/perf_test_long_cycles_nbrows_cycle_length_41000_80.log.diff
	 tail -10  logs/perf_test_long_cycles_nbrows_cycle_length_41000_80.log.diff 

perf/test_ozone_ar_speed.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_ar_speed.py  >  logs/perf_test_ozone_ar_speed.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_ozone_ar_speed.log logs/perf_test_ozone_ar_speed.log  >  logs/perf_test_ozone_ar_speed.log.diff
	 tail -10  logs/perf_test_ozone_ar_speed.log.diff 

perf/test_ozone_ar_speed_many.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_ar_speed_many.py  >  logs/perf_test_ozone_ar_speed_many.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_ozone_ar_speed_many.log logs/perf_test_ozone_ar_speed_many.log  >  logs/perf_test_ozone_ar_speed_many.log.diff
	 tail -10  logs/perf_test_ozone_ar_speed_many.log.diff 

perf/test_ozone_ar_speed_order_0.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_ar_speed_order_0.py  >  logs/perf_test_ozone_ar_speed_order_0.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_ozone_ar_speed_order_0.log logs/perf_test_ozone_ar_speed_order_0.log  >  logs/perf_test_ozone_ar_speed_order_0.log.diff
	 tail -10  logs/perf_test_ozone_ar_speed_order_0.log.diff 

perf/test_ozone_ar_speed_order_100.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_ar_speed_order_100.py  >  logs/perf_test_ozone_ar_speed_order_100.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_ozone_ar_speed_order_100.log logs/perf_test_ozone_ar_speed_order_100.log  >  logs/perf_test_ozone_ar_speed_order_100.log.diff
	 tail -10  logs/perf_test_ozone_ar_speed_order_100.log.diff 

perf/test_ozone_ar_speed_order_150.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_ar_speed_order_150.py  >  logs/perf_test_ozone_ar_speed_order_150.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_ozone_ar_speed_order_150.log logs/perf_test_ozone_ar_speed_order_150.log  >  logs/perf_test_ozone_ar_speed_order_150.log.diff
	 tail -10  logs/perf_test_ozone_ar_speed_order_150.log.diff 

perf/test_ozone_ar_speed_order_200.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_ar_speed_order_200.py  >  logs/perf_test_ozone_ar_speed_order_200.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_ozone_ar_speed_order_200.log logs/perf_test_ozone_ar_speed_order_200.log  >  logs/perf_test_ozone_ar_speed_order_200.log.diff
	 tail -10  logs/perf_test_ozone_ar_speed_order_200.log.diff 

perf/test_ozone_ar_speed_order_250.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_ar_speed_order_250.py  >  logs/perf_test_ozone_ar_speed_order_250.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_ozone_ar_speed_order_250.log logs/perf_test_ozone_ar_speed_order_250.log  >  logs/perf_test_ozone_ar_speed_order_250.log.diff
	 tail -10  logs/perf_test_ozone_ar_speed_order_250.log.diff 

perf/test_ozone_ar_speed_order_300.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_ar_speed_order_300.py  >  logs/perf_test_ozone_ar_speed_order_300.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_ozone_ar_speed_order_300.log logs/perf_test_ozone_ar_speed_order_300.log  >  logs/perf_test_ozone_ar_speed_order_300.log.diff
	 tail -10  logs/perf_test_ozone_ar_speed_order_300.log.diff 

perf/test_ozone_ar_speed_order_350.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_ar_speed_order_350.py  >  logs/perf_test_ozone_ar_speed_order_350.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_ozone_ar_speed_order_350.log logs/perf_test_ozone_ar_speed_order_350.log  >  logs/perf_test_ozone_ar_speed_order_350.log.diff
	 tail -10  logs/perf_test_ozone_ar_speed_order_350.log.diff 

perf/test_ozone_ar_speed_order_400.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_ar_speed_order_400.py  >  logs/perf_test_ozone_ar_speed_order_400.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_ozone_ar_speed_order_400.log logs/perf_test_ozone_ar_speed_order_400.log  >  logs/perf_test_ozone_ar_speed_order_400.log.diff
	 tail -10  logs/perf_test_ozone_ar_speed_order_400.log.diff 

perf/test_ozone_ar_speed_order_450.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_ar_speed_order_450.py  >  logs/perf_test_ozone_ar_speed_order_450.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_ozone_ar_speed_order_450.log logs/perf_test_ozone_ar_speed_order_450.log  >  logs/perf_test_ozone_ar_speed_order_450.log.diff
	 tail -10  logs/perf_test_ozone_ar_speed_order_450.log.diff 

perf/test_ozone_ar_speed_order_50.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_ar_speed_order_50.py  >  logs/perf_test_ozone_ar_speed_order_50.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_ozone_ar_speed_order_50.log logs/perf_test_ozone_ar_speed_order_50.log  >  logs/perf_test_ozone_ar_speed_order_50.log.diff
	 tail -10  logs/perf_test_ozone_ar_speed_order_50.log.diff 

perf/test_ozone_ar_speed_order_500.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_ar_speed_order_500.py  >  logs/perf_test_ozone_ar_speed_order_500.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_ozone_ar_speed_order_500.log logs/perf_test_ozone_ar_speed_order_500.log  >  logs/perf_test_ozone_ar_speed_order_500.log.diff
	 tail -10  logs/perf_test_ozone_ar_speed_order_500.log.diff 

perf/test_ozone_ar_speed_order_550.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_ar_speed_order_550.py  >  logs/perf_test_ozone_ar_speed_order_550.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_ozone_ar_speed_order_550.log logs/perf_test_ozone_ar_speed_order_550.log  >  logs/perf_test_ozone_ar_speed_order_550.log.diff
	 tail -10  logs/perf_test_ozone_ar_speed_order_550.log.diff 

perf/test_ozone_ar_speed_order_600.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_ar_speed_order_600.py  >  logs/perf_test_ozone_ar_speed_order_600.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_ozone_ar_speed_order_600.log logs/perf_test_ozone_ar_speed_order_600.log  >  logs/perf_test_ozone_ar_speed_order_600.log.diff
	 tail -10  logs/perf_test_ozone_ar_speed_order_600.log.diff 

perf/test_ozone_ar_speed_order_650.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_ar_speed_order_650.py  >  logs/perf_test_ozone_ar_speed_order_650.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_ozone_ar_speed_order_650.log logs/perf_test_ozone_ar_speed_order_650.log  >  logs/perf_test_ozone_ar_speed_order_650.log.diff
	 tail -10  logs/perf_test_ozone_ar_speed_order_650.log.diff 

perf/test_ozone_ar_speed_order_700.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_ar_speed_order_700.py  >  logs/perf_test_ozone_ar_speed_order_700.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_ozone_ar_speed_order_700.log logs/perf_test_ozone_ar_speed_order_700.log  >  logs/perf_test_ozone_ar_speed_order_700.log.diff
	 tail -10  logs/perf_test_ozone_ar_speed_order_700.log.diff 

perf/test_ozone_ar_speed_order_750.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_ar_speed_order_750.py  >  logs/perf_test_ozone_ar_speed_order_750.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_ozone_ar_speed_order_750.log logs/perf_test_ozone_ar_speed_order_750.log  >  logs/perf_test_ozone_ar_speed_order_750.log.diff
	 tail -10  logs/perf_test_ozone_ar_speed_order_750.log.diff 

perf/test_ozone_ar_speed_order_800.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_ar_speed_order_800.py  >  logs/perf_test_ozone_ar_speed_order_800.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_ozone_ar_speed_order_800.log logs/perf_test_ozone_ar_speed_order_800.log  >  logs/perf_test_ozone_ar_speed_order_800.log.diff
	 tail -10  logs/perf_test_ozone_ar_speed_order_800.log.diff 

perf/test_ozone_ar_speed_order_850.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_ar_speed_order_850.py  >  logs/perf_test_ozone_ar_speed_order_850.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_ozone_ar_speed_order_850.log logs/perf_test_ozone_ar_speed_order_850.log  >  logs/perf_test_ozone_ar_speed_order_850.log.diff
	 tail -10  logs/perf_test_ozone_ar_speed_order_850.log.diff 

perf/test_ozone_ar_speed_order_900.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_ar_speed_order_900.py  >  logs/perf_test_ozone_ar_speed_order_900.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_ozone_ar_speed_order_900.log logs/perf_test_ozone_ar_speed_order_900.log  >  logs/perf_test_ozone_ar_speed_order_900.log.diff
	 tail -10  logs/perf_test_ozone_ar_speed_order_900.log.diff 

perf/test_ozone_ar_speed_order_950.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_ar_speed_order_950.py  >  logs/perf_test_ozone_ar_speed_order_950.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_ozone_ar_speed_order_950.log logs/perf_test_ozone_ar_speed_order_950.log  >  logs/perf_test_ozone_ar_speed_order_950.log.diff
	 tail -10  logs/perf_test_ozone_ar_speed_order_950.log.diff 

perf/test_ozone_debug_perf.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_debug_perf.py  >  logs/perf_test_ozone_debug_perf.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_ozone_debug_perf.log logs/perf_test_ozone_debug_perf.log  >  logs/perf_test_ozone_debug_perf.log.diff
	 tail -10  logs/perf_test_ozone_debug_perf.log.diff 

perf/test_ozone_long_series.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_long_series.py  >  logs/perf_test_ozone_long_series.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_ozone_long_series.log logs/perf_test_ozone_long_series.log  >  logs/perf_test_ozone_long_series.log.diff
	 tail -10  logs/perf_test_ozone_long_series.log.diff 

perf/test_ozone_long_series_2.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_long_series_2.py  >  logs/perf_test_ozone_long_series_2.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_ozone_long_series_2.log logs/perf_test_ozone_long_series_2.log  >  logs/perf_test_ozone_long_series_2.log.diff
	 tail -10  logs/perf_test_ozone_long_series_2.log.diff 

perf/test_parallel.py  :  
	 -$(PYTHON)  tests/perf/test_parallel.py  >  logs/perf_test_parallel.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_parallel.log logs/perf_test_parallel.log  >  logs/perf_test_parallel.log.diff
	 tail -10  logs/perf_test_parallel.log.diff 

perf/test_perf1.py  :  
	 -$(PYTHON)  tests/perf/test_perf1.py  >  logs/perf_test_perf1.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_perf1.log logs/perf_test_perf1.log  >  logs/perf_test_perf1.log.diff
	 tail -10  logs/perf_test_perf1.log.diff 

perf/test_perf2.py  :  
	 -$(PYTHON)  tests/perf/test_perf2.py  >  logs/perf_test_perf2.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_perf2.log logs/perf_test_perf2.log  >  logs/perf_test_perf2.log.diff
	 tail -10  logs/perf_test_perf2.log.diff 

perf/test_web-traffic-time-series-forecasting.py  :  
	 -$(PYTHON)  tests/perf/test_web-traffic-time-series-forecasting.py  >  logs/perf_test_web-traffic-time-series-forecasting.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_web-traffic-time-series-forecasting.log logs/perf_test_web-traffic-time-series-forecasting.log  >  logs/perf_test_web-traffic-time-series-forecasting.log.diff
	 tail -10  logs/perf_test_web-traffic-time-series-forecasting.log.diff 

perf/test_web-traffic-time-series-forecasting_all.py  :  
	 -$(PYTHON)  tests/perf/test_web-traffic-time-series-forecasting_all.py  >  logs/perf_test_web-traffic-time-series-forecasting_all.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf/test_web-traffic-time-series-forecasting_all.log logs/perf_test_web-traffic-time-series-forecasting_all.log  >  logs/perf_test_web-traffic-time-series-forecasting_all.log.diff
	 tail -10  logs/perf_test_web-traffic-time-series-forecasting_all.log.diff 



 perf :  perf/test_web-traffic-time-series-forecasting_all.py perf/test_web-traffic-time-series-forecasting.py perf/test_perf2.py perf/test_perf1.py perf/test_parallel.py perf/test_ozone_long_series_2.py perf/test_ozone_long_series.py perf/test_ozone_debug_perf.py perf/test_ozone_ar_speed_order_950.py perf/test_ozone_ar_speed_order_900.py perf/test_ozone_ar_speed_order_850.py perf/test_ozone_ar_speed_order_800.py perf/test_ozone_ar_speed_order_750.py perf/test_ozone_ar_speed_order_700.py perf/test_ozone_ar_speed_order_650.py perf/test_ozone_ar_speed_order_600.py perf/test_ozone_ar_speed_order_550.py perf/test_ozone_ar_speed_order_500.py perf/test_ozone_ar_speed_order_50.py perf/test_ozone_ar_speed_order_450.py perf/test_ozone_ar_speed_order_400.py perf/test_ozone_ar_speed_order_350.py perf/test_ozone_ar_speed_order_300.py perf/test_ozone_ar_speed_order_250.py perf/test_ozone_ar_speed_order_200.py perf/test_ozone_ar_speed_order_150.py perf/test_ozone_ar_speed_order_100.py perf/test_ozone_ar_speed_order_0.py perf/test_ozone_ar_speed_many.py perf/test_ozone_ar_speed.py perf/test_long_cycles_nbrows_cycle_length_41000_80.py perf/test_long_cycles_nbrows_cycle_length_41000_440.py perf/test_long_cycles_nbrows_cycle_length_41000_380.py perf/test_long_cycles_nbrows_cycle_length_41000_320.py perf/test_long_cycles_nbrows_cycle_length_41000_260.py perf/test_long_cycles_nbrows_cycle_length_41000_200.py perf/test_long_cycles_nbrows_cycle_length_41000_20.py perf/test_long_cycles_nbrows_cycle_length_41000_140.py perf/test_long_cycles_nbrows_cycle_length_31000_80.py perf/test_long_cycles_nbrows_cycle_length_31000_440.py perf/test_long_cycles_nbrows_cycle_length_31000_380.py perf/test_long_cycles_nbrows_cycle_length_31000_320.py perf/test_long_cycles_nbrows_cycle_length_31000_260.py perf/test_long_cycles_nbrows_cycle_length_31000_200.py perf/test_long_cycles_nbrows_cycle_length_31000_20.py perf/test_long_cycles_nbrows_cycle_length_31000_140.py perf/test_long_cycles_nbrows_cycle_length_21000_80.py perf/test_long_cycles_nbrows_cycle_length_21000_440.py perf/test_long_cycles_nbrows_cycle_length_21000_380.py perf/test_long_cycles_nbrows_cycle_length_21000_320.py perf/test_long_cycles_nbrows_cycle_length_21000_260.py perf/test_long_cycles_nbrows_cycle_length_21000_200.py perf/test_long_cycles_nbrows_cycle_length_21000_20.py perf/test_long_cycles_nbrows_cycle_length_21000_140.py perf/test_long_cycles_nbrows_cycle_length_11000_80.py perf/test_long_cycles_nbrows_cycle_length_11000_440.py perf/test_long_cycles_nbrows_cycle_length_11000_380.py perf/test_long_cycles_nbrows_cycle_length_11000_320.py perf/test_long_cycles_nbrows_cycle_length_11000_260.py perf/test_long_cycles_nbrows_cycle_length_11000_200.py perf/test_long_cycles_nbrows_cycle_length_11000_20.py perf/test_long_cycles_nbrows_cycle_length_11000_140.py perf/test_long_cycles_nbrows_cycle_length_1000_80.py perf/test_long_cycles_nbrows_cycle_length_1000_440.py perf/test_long_cycles_nbrows_cycle_length_1000_380.py perf/test_long_cycles_nbrows_cycle_length_1000_320.py perf/test_long_cycles_nbrows_cycle_length_1000_260.py perf/test_long_cycles_nbrows_cycle_length_1000_200.py perf/test_long_cycles_nbrows_cycle_length_1000_20.py perf/test_long_cycles_nbrows_cycle_length_1000_140.py perf/test_cycles_full_long_long.py perf/test_cycles_full_long.py perf/gen_long_cycles_tests.py perf/gen_ar_speed_tests.py  
 

perf_LnQ/test_air_passengers_LnQ.py  :  
	 -$(PYTHON)  tests/perf_LnQ/test_air_passengers_LnQ.py  >  logs/perf_LnQ_test_air_passengers_LnQ.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf_LnQ/test_air_passengers_LnQ.log logs/perf_LnQ_test_air_passengers_LnQ.log  >  logs/perf_LnQ_test_air_passengers_LnQ.log.diff
	 tail -10  logs/perf_LnQ_test_air_passengers_LnQ.log.diff 

perf_LnQ/test_ozone_LnQ.py  :  
	 -$(PYTHON)  tests/perf_LnQ/test_ozone_LnQ.py  >  logs/perf_LnQ_test_ozone_LnQ.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf_LnQ/test_ozone_LnQ.log logs/perf_LnQ_test_ozone_LnQ.log  >  logs/perf_LnQ_test_ozone_LnQ.log.diff
	 tail -10  logs/perf_LnQ_test_ozone_LnQ.log.diff 



 perf_LnQ :  perf_LnQ/test_ozone_LnQ.py perf_LnQ/test_air_passengers_LnQ.py  
 

perf_MASE_RMSSE/user_test_MAPE.py  :  
	 -$(PYTHON)  tests/perf_MASE_RMSSE/user_test_MAPE.py  >  logs/perf_MASE_RMSSE_user_test_MAPE.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf_MASE_RMSSE/user_test_MAPE.log logs/perf_MASE_RMSSE_user_test_MAPE.log  >  logs/perf_MASE_RMSSE_user_test_MAPE.log.diff
	 tail -10  logs/perf_MASE_RMSSE_user_test_MAPE.log.diff 

perf_MASE_RMSSE/user_test_MASE.py  :  
	 -$(PYTHON)  tests/perf_MASE_RMSSE/user_test_MASE.py  >  logs/perf_MASE_RMSSE_user_test_MASE.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf_MASE_RMSSE/user_test_MASE.log logs/perf_MASE_RMSSE_user_test_MASE.log  >  logs/perf_MASE_RMSSE_user_test_MASE.log.diff
	 tail -10  logs/perf_MASE_RMSSE_user_test_MASE.log.diff 

perf_MASE_RMSSE/user_test_RMSE.py  :  
	 -$(PYTHON)  tests/perf_MASE_RMSSE/user_test_RMSE.py  >  logs/perf_MASE_RMSSE_user_test_RMSE.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf_MASE_RMSSE/user_test_RMSE.log logs/perf_MASE_RMSSE_user_test_RMSE.log  >  logs/perf_MASE_RMSSE_user_test_RMSE.log.diff
	 tail -10  logs/perf_MASE_RMSSE_user_test_RMSE.log.diff 

perf_MASE_RMSSE/user_test_RMSSE.py  :  
	 -$(PYTHON)  tests/perf_MASE_RMSSE/user_test_RMSSE.py  >  logs/perf_MASE_RMSSE_user_test_RMSSE.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf_MASE_RMSSE/user_test_RMSSE.log logs/perf_MASE_RMSSE_user_test_RMSSE.log  >  logs/perf_MASE_RMSSE_user_test_RMSSE.log.diff
	 tail -10  logs/perf_MASE_RMSSE_user_test_RMSSE.log.diff 



 perf_MASE_RMSSE :  perf_MASE_RMSSE/user_test_RMSSE.py perf_MASE_RMSSE/user_test_RMSE.py perf_MASE_RMSSE/user_test_MASE.py perf_MASE_RMSSE/user_test_MAPE.py  
 

perf_MedAE/test_air_passengers_MedAE.py  :  
	 -$(PYTHON)  tests/perf_MedAE/test_air_passengers_MedAE.py  >  logs/perf_MedAE_test_air_passengers_MedAE.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf_MedAE/test_air_passengers_MedAE.log logs/perf_MedAE_test_air_passengers_MedAE.log  >  logs/perf_MedAE_test_air_passengers_MedAE.log.diff
	 tail -10  logs/perf_MedAE_test_air_passengers_MedAE.log.diff 

perf_MedAE/test_ozone_MedAE.py  :  
	 -$(PYTHON)  tests/perf_MedAE/test_ozone_MedAE.py  >  logs/perf_MedAE_test_ozone_MedAE.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perf_MedAE/test_ozone_MedAE.log logs/perf_MedAE_test_ozone_MedAE.log  >  logs/perf_MedAE_test_ozone_MedAE.log.diff
	 tail -10  logs/perf_MedAE_test_ozone_MedAE.log.diff 



 perf_MedAE :  perf_MedAE/test_ozone_MedAE.py perf_MedAE/test_air_passengers_MedAE.py  
 

perfs/test_ozone_perf_measure.py  :  
	 -$(PYTHON)  tests/perfs/test_ozone_perf_measure.py  >  logs/perfs_test_ozone_perf_measure.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perfs/test_ozone_perf_measure.log logs/perfs_test_ozone_perf_measure.log  >  logs/perfs_test_ozone_perf_measure.log.diff
	 tail -10  logs/perfs_test_ozone_perf_measure.log.diff 

perfs/test_ozone_perf_measure_L1.py  :  
	 -$(PYTHON)  tests/perfs/test_ozone_perf_measure_L1.py  >  logs/perfs_test_ozone_perf_measure_L1.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perfs/test_ozone_perf_measure_L1.log logs/perfs_test_ozone_perf_measure_L1.log  >  logs/perfs_test_ozone_perf_measure_L1.log.diff
	 tail -10  logs/perfs_test_ozone_perf_measure_L1.log.diff 

perfs/test_ozone_perf_measure_L2.py  :  
	 -$(PYTHON)  tests/perfs/test_ozone_perf_measure_L2.py  >  logs/perfs_test_ozone_perf_measure_L2.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perfs/test_ozone_perf_measure_L2.log logs/perfs_test_ozone_perf_measure_L2.log  >  logs/perfs_test_ozone_perf_measure_L2.log.diff
	 tail -10  logs/perfs_test_ozone_perf_measure_L2.log.diff 

perfs/test_ozone_perf_measure_MAPE.py  :  
	 -$(PYTHON)  tests/perfs/test_ozone_perf_measure_MAPE.py  >  logs/perfs_test_ozone_perf_measure_MAPE.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perfs/test_ozone_perf_measure_MAPE.log logs/perfs_test_ozone_perf_measure_MAPE.log  >  logs/perfs_test_ozone_perf_measure_MAPE.log.diff
	 tail -10  logs/perfs_test_ozone_perf_measure_MAPE.log.diff 

perfs/test_ozone_perf_measure_MASE.py  :  
	 -$(PYTHON)  tests/perfs/test_ozone_perf_measure_MASE.py  >  logs/perfs_test_ozone_perf_measure_MASE.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/perfs/test_ozone_perf_measure_MASE.log logs/perfs_test_ozone_perf_measure_MASE.log  >  logs/perfs_test_ozone_perf_measure_MASE.log.diff
	 tail -10  logs/perfs_test_ozone_perf_measure_MASE.log.diff 



 perfs :  perfs/test_ozone_perf_measure_MASE.py perfs/test_ozone_perf_measure_MAPE.py perfs/test_ozone_perf_measure_L2.py perfs/test_ozone_perf_measure_L1.py perfs/test_ozone_perf_measure.py  
 

plots/test_air_passengers_plots.py  :  
	 -$(PYTHON)  tests/plots/test_air_passengers_plots.py  >  logs/plots_test_air_passengers_plots.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/plots/test_air_passengers_plots.log logs/plots_test_air_passengers_plots.log  >  logs/plots_test_air_passengers_plots.log.diff
	 tail -10  logs/plots_test_air_passengers_plots.log.diff 

plots/test_hierarchy_AU_all_reconciliations_plots.py  :  
	 -$(PYTHON)  tests/plots/test_hierarchy_AU_all_reconciliations_plots.py  >  logs/plots_test_hierarchy_AU_all_reconciliations_plots.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/plots/test_hierarchy_AU_all_reconciliations_plots.log logs/plots_test_hierarchy_AU_all_reconciliations_plots.log  >  logs/plots_test_hierarchy_AU_all_reconciliations_plots.log.diff
	 tail -10  logs/plots_test_hierarchy_AU_all_reconciliations_plots.log.diff 

plots/test_hierarchy_AU_plots.py  :  
	 -$(PYTHON)  tests/plots/test_hierarchy_AU_plots.py  >  logs/plots_test_hierarchy_AU_plots.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/plots/test_hierarchy_AU_plots.log logs/plots_test_hierarchy_AU_plots.log  >  logs/plots_test_hierarchy_AU_plots.log.diff
	 tail -10  logs/plots_test_hierarchy_AU_plots.log.diff 

plots/test_ozone_plots.py  :  
	 -$(PYTHON)  tests/plots/test_ozone_plots.py  >  logs/plots_test_ozone_plots.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/plots/test_ozone_plots.log logs/plots_test_ozone_plots.log  >  logs/plots_test_ozone_plots.log.diff
	 tail -10  logs/plots_test_ozone_plots.log.diff 



 plots :  plots/test_ozone_plots.py plots/test_hierarchy_AU_plots.py plots/test_hierarchy_AU_all_reconciliations_plots.py plots/test_air_passengers_plots.py  
 

probabilistic_forecasting/test_air_passengers.py  :  
	 -$(PYTHON)  tests/probabilistic_forecasting/test_air_passengers.py  >  logs/probabilistic_forecasting_test_air_passengers.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/probabilistic_forecasting/test_air_passengers.log logs/probabilistic_forecasting_test_air_passengers.log  >  logs/probabilistic_forecasting_test_air_passengers.log.diff
	 tail -10  logs/probabilistic_forecasting_test_air_passengers.log.diff 

probabilistic_forecasting/test_ozone.py  :  
	 -$(PYTHON)  tests/probabilistic_forecasting/test_ozone.py  >  logs/probabilistic_forecasting_test_ozone.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/probabilistic_forecasting/test_ozone.log logs/probabilistic_forecasting_test_ozone.log  >  logs/probabilistic_forecasting_test_ozone.log.diff
	 tail -10  logs/probabilistic_forecasting_test_ozone.log.diff 



 probabilistic_forecasting :  probabilistic_forecasting/test_ozone.py probabilistic_forecasting/test_air_passengers.py  
 

pytorch/test_ozone_exogenous_LSTMX_keras.py  :  
	 -$(PYTHON)  tests/pytorch/test_ozone_exogenous_LSTMX_keras.py  >  logs/pytorch_test_ozone_exogenous_LSTMX_keras.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/pytorch/test_ozone_exogenous_LSTMX_keras.log logs/pytorch_test_ozone_exogenous_LSTMX_keras.log  >  logs/pytorch_test_ozone_exogenous_LSTMX_keras.log.diff
	 tail -10  logs/pytorch_test_ozone_exogenous_LSTMX_keras.log.diff 

pytorch/test_ozone_exogenous_LSTMX_pytorch.py  :  
	 -$(PYTHON)  tests/pytorch/test_ozone_exogenous_LSTMX_pytorch.py  >  logs/pytorch_test_ozone_exogenous_LSTMX_pytorch.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/pytorch/test_ozone_exogenous_LSTMX_pytorch.log logs/pytorch_test_ozone_exogenous_LSTMX_pytorch.log  >  logs/pytorch_test_ozone_exogenous_LSTMX_pytorch.log.diff
	 tail -10  logs/pytorch_test_ozone_exogenous_LSTMX_pytorch.log.diff 

pytorch/test_ozone_exogenous_LSTMX_pytorch_or_keras.py  :  
	 -$(PYTHON)  tests/pytorch/test_ozone_exogenous_LSTMX_pytorch_or_keras.py  >  logs/pytorch_test_ozone_exogenous_LSTMX_pytorch_or_keras.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/pytorch/test_ozone_exogenous_LSTMX_pytorch_or_keras.log logs/pytorch_test_ozone_exogenous_LSTMX_pytorch_or_keras.log  >  logs/pytorch_test_ozone_exogenous_LSTMX_pytorch_or_keras.log.diff
	 tail -10  logs/pytorch_test_ozone_exogenous_LSTMX_pytorch_or_keras.log.diff 

pytorch/test_ozone_exogenous_MLPX_keras.py  :  
	 -$(PYTHON)  tests/pytorch/test_ozone_exogenous_MLPX_keras.py  >  logs/pytorch_test_ozone_exogenous_MLPX_keras.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/pytorch/test_ozone_exogenous_MLPX_keras.log logs/pytorch_test_ozone_exogenous_MLPX_keras.log  >  logs/pytorch_test_ozone_exogenous_MLPX_keras.log.diff
	 tail -10  logs/pytorch_test_ozone_exogenous_MLPX_keras.log.diff 

pytorch/test_ozone_exogenous_MLPX_pytorch.py  :  
	 -$(PYTHON)  tests/pytorch/test_ozone_exogenous_MLPX_pytorch.py  >  logs/pytorch_test_ozone_exogenous_MLPX_pytorch.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/pytorch/test_ozone_exogenous_MLPX_pytorch.log logs/pytorch_test_ozone_exogenous_MLPX_pytorch.log  >  logs/pytorch_test_ozone_exogenous_MLPX_pytorch.log.diff
	 tail -10  logs/pytorch_test_ozone_exogenous_MLPX_pytorch.log.diff 

pytorch/test_ozone_exogenous_MLPX_pytorch_or_keras.py  :  
	 -$(PYTHON)  tests/pytorch/test_ozone_exogenous_MLPX_pytorch_or_keras.py  >  logs/pytorch_test_ozone_exogenous_MLPX_pytorch_or_keras.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/pytorch/test_ozone_exogenous_MLPX_pytorch_or_keras.log logs/pytorch_test_ozone_exogenous_MLPX_pytorch_or_keras.log  >  logs/pytorch_test_ozone_exogenous_MLPX_pytorch_or_keras.log.diff
	 tail -10  logs/pytorch_test_ozone_exogenous_MLPX_pytorch_or_keras.log.diff 

pytorch/test_ozone_exogenous_gen.py  :  
	 -$(PYTHON)  tests/pytorch/test_ozone_exogenous_gen.py  >  logs/pytorch_test_ozone_exogenous_gen.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/pytorch/test_ozone_exogenous_gen.log logs/pytorch_test_ozone_exogenous_gen.log  >  logs/pytorch_test_ozone_exogenous_gen.log.diff
	 tail -10  logs/pytorch_test_ozone_exogenous_gen.log.diff 

pytorch/test_ozone_exogenous_keras_256_threads.py  :  
	 -$(PYTHON)  tests/pytorch/test_ozone_exogenous_keras_256_threads.py  >  logs/pytorch_test_ozone_exogenous_keras_256_threads.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/pytorch/test_ozone_exogenous_keras_256_threads.log logs/pytorch_test_ozone_exogenous_keras_256_threads.log  >  logs/pytorch_test_ozone_exogenous_keras_256_threads.log.diff
	 tail -10  logs/pytorch_test_ozone_exogenous_keras_256_threads.log.diff 

pytorch/test_ozone_exogenous_pytorch_256_threads.py  :  
	 -$(PYTHON)  tests/pytorch/test_ozone_exogenous_pytorch_256_threads.py  >  logs/pytorch_test_ozone_exogenous_pytorch_256_threads.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/pytorch/test_ozone_exogenous_pytorch_256_threads.log logs/pytorch_test_ozone_exogenous_pytorch_256_threads.log  >  logs/pytorch_test_ozone_exogenous_pytorch_256_threads.log.diff
	 tail -10  logs/pytorch_test_ozone_exogenous_pytorch_256_threads.log.diff 



 pytorch :  pytorch/test_ozone_exogenous_pytorch_256_threads.py pytorch/test_ozone_exogenous_keras_256_threads.py pytorch/test_ozone_exogenous_gen.py pytorch/test_ozone_exogenous_MLPX_pytorch_or_keras.py pytorch/test_ozone_exogenous_MLPX_pytorch.py pytorch/test_ozone_exogenous_MLPX_keras.py pytorch/test_ozone_exogenous_LSTMX_pytorch_or_keras.py pytorch/test_ozone_exogenous_LSTMX_pytorch.py pytorch/test_ozone_exogenous_LSTMX_keras.py  
 

real-life/test_sof_example.py  :  
	 -$(PYTHON)  tests/real-life/test_sof_example.py  >  logs/real-life_test_sof_example.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/real-life/test_sof_example.log logs/real-life_test_sof_example.log  >  logs/real-life_test_sof_example.log.diff
	 tail -10  logs/real-life_test_sof_example.log.diff 

real-life/test_sof_example2.py  :  
	 -$(PYTHON)  tests/real-life/test_sof_example2.py  >  logs/real-life_test_sof_example2.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/real-life/test_sof_example2.log logs/real-life_test_sof_example2.log  >  logs/real-life_test_sof_example2.log.diff
	 tail -10  logs/real-life_test_sof_example2.log.diff 



 real-life :  real-life/test_sof_example2.py real-life/test_sof_example.py  
 

sampling/test_ozone_with_no_sampling.py  :  
	 -$(PYTHON)  tests/sampling/test_ozone_with_no_sampling.py  >  logs/sampling_test_ozone_with_no_sampling.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/sampling/test_ozone_with_no_sampling.log logs/sampling_test_ozone_with_no_sampling.log  >  logs/sampling_test_ozone_with_no_sampling.log.diff
	 tail -10  logs/sampling_test_ozone_with_no_sampling.log.diff 

sampling/test_ozone_with_sampling.py  :  
	 -$(PYTHON)  tests/sampling/test_ozone_with_sampling.py  >  logs/sampling_test_ozone_with_sampling.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/sampling/test_ozone_with_sampling.log logs/sampling_test_ozone_with_sampling.log  >  logs/sampling_test_ozone_with_sampling.log.diff
	 tail -10  logs/sampling_test_ozone_with_sampling.log.diff 

sampling/test_ozone_with_sampling_2.py  :  
	 -$(PYTHON)  tests/sampling/test_ozone_with_sampling_2.py  >  logs/sampling_test_ozone_with_sampling_2.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/sampling/test_ozone_with_sampling_2.log logs/sampling_test_ozone_with_sampling_2.log  >  logs/sampling_test_ozone_with_sampling_2.log.diff
	 tail -10  logs/sampling_test_ozone_with_sampling_2.log.diff 

sampling/test_ozone_with_sampling_3.py  :  
	 -$(PYTHON)  tests/sampling/test_ozone_with_sampling_3.py  >  logs/sampling_test_ozone_with_sampling_3.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/sampling/test_ozone_with_sampling_3.log logs/sampling_test_ozone_with_sampling_3.log  >  logs/sampling_test_ozone_with_sampling_3.log.diff
	 tail -10  logs/sampling_test_ozone_with_sampling_3.log.diff 



 sampling :  sampling/test_ozone_with_sampling_3.py sampling/test_ozone_with_sampling_2.py sampling/test_ozone_with_sampling.py sampling/test_ozone_with_no_sampling.py  
 

svr/test_air_passengers_svr.py  :  
	 -$(PYTHON)  tests/svr/test_air_passengers_svr.py  >  logs/svr_test_air_passengers_svr.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/svr/test_air_passengers_svr.log logs/svr_test_air_passengers_svr.log  >  logs/svr_test_air_passengers_svr.log.diff
	 tail -10  logs/svr_test_air_passengers_svr.log.diff 

svr/test_air_passengers_svr_only.py  :  
	 -$(PYTHON)  tests/svr/test_air_passengers_svr_only.py  >  logs/svr_test_air_passengers_svr_only.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/svr/test_air_passengers_svr_only.log logs/svr_test_air_passengers_svr_only.log  >  logs/svr_test_air_passengers_svr_only.log.diff
	 tail -10  logs/svr_test_air_passengers_svr_only.log.diff 

svr/test_ozone_svr.py  :  
	 -$(PYTHON)  tests/svr/test_ozone_svr.py  >  logs/svr_test_ozone_svr.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/svr/test_ozone_svr.log logs/svr_test_ozone_svr.log  >  logs/svr_test_ozone_svr.log.diff
	 tail -10  logs/svr_test_ozone_svr.log.diff 

svr/test_ozone_svr_only.py  :  
	 -$(PYTHON)  tests/svr/test_ozone_svr_only.py  >  logs/svr_test_ozone_svr_only.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/svr/test_ozone_svr_only.log logs/svr_test_ozone_svr_only.log  >  logs/svr_test_ozone_svr_only.log.diff
	 tail -10  logs/svr_test_ozone_svr_only.log.diff 



 svr :  svr/test_ozone_svr_only.py svr/test_ozone_svr.py svr/test_air_passengers_svr_only.py svr/test_air_passengers_svr.py  
 

temporal_hierarchy/test_temporal_demo_1.py  :  
	 -$(PYTHON)  tests/temporal_hierarchy/test_temporal_demo_1.py  >  logs/temporal_hierarchy_test_temporal_demo_1.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/temporal_hierarchy/test_temporal_demo_1.log logs/temporal_hierarchy_test_temporal_demo_1.log  >  logs/temporal_hierarchy_test_temporal_demo_1.log.diff
	 tail -10  logs/temporal_hierarchy_test_temporal_demo_1.log.diff 

temporal_hierarchy/test_temporal_demo_daily_D_W_2W.py  :  
	 -$(PYTHON)  tests/temporal_hierarchy/test_temporal_demo_daily_D_W_2W.py  >  logs/temporal_hierarchy_test_temporal_demo_daily_D_W_2W.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/temporal_hierarchy/test_temporal_demo_daily_D_W_2W.log logs/temporal_hierarchy_test_temporal_demo_daily_D_W_2W.log  >  logs/temporal_hierarchy_test_temporal_demo_daily_D_W_2W.log.diff
	 tail -10  logs/temporal_hierarchy_test_temporal_demo_daily_D_W_2W.log.diff 

temporal_hierarchy/test_temporal_demo_daily_D_W_2W_Q.py  :  
	 -$(PYTHON)  tests/temporal_hierarchy/test_temporal_demo_daily_D_W_2W_Q.py  >  logs/temporal_hierarchy_test_temporal_demo_daily_D_W_2W_Q.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/temporal_hierarchy/test_temporal_demo_daily_D_W_2W_Q.log logs/temporal_hierarchy_test_temporal_demo_daily_D_W_2W_Q.log  >  logs/temporal_hierarchy_test_temporal_demo_daily_D_W_2W_Q.log.diff
	 tail -10  logs/temporal_hierarchy_test_temporal_demo_daily_D_W_2W_Q.log.diff 

temporal_hierarchy/test_temporal_demo_daily_D_W_M.py  :  
	 -$(PYTHON)  tests/temporal_hierarchy/test_temporal_demo_daily_D_W_M.py  >  logs/temporal_hierarchy_test_temporal_demo_daily_D_W_M.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/temporal_hierarchy/test_temporal_demo_daily_D_W_M.log logs/temporal_hierarchy_test_temporal_demo_daily_D_W_M.log  >  logs/temporal_hierarchy_test_temporal_demo_daily_D_W_M.log.diff
	 tail -10  logs/temporal_hierarchy_test_temporal_demo_daily_D_W_M.log.diff 

temporal_hierarchy/test_temporal_demo_daily_D_W_M_Q.py  :  
	 -$(PYTHON)  tests/temporal_hierarchy/test_temporal_demo_daily_D_W_M_Q.py  >  logs/temporal_hierarchy_test_temporal_demo_daily_D_W_M_Q.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/temporal_hierarchy/test_temporal_demo_daily_D_W_M_Q.log logs/temporal_hierarchy_test_temporal_demo_daily_D_W_M_Q.log  >  logs/temporal_hierarchy_test_temporal_demo_daily_D_W_M_Q.log.diff
	 tail -10  logs/temporal_hierarchy_test_temporal_demo_daily_D_W_M_Q.log.diff 

temporal_hierarchy/test_temporal_demo_daily_D_W_Q.py  :  
	 -$(PYTHON)  tests/temporal_hierarchy/test_temporal_demo_daily_D_W_Q.py  >  logs/temporal_hierarchy_test_temporal_demo_daily_D_W_Q.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/temporal_hierarchy/test_temporal_demo_daily_D_W_Q.log logs/temporal_hierarchy_test_temporal_demo_daily_D_W_Q.log  >  logs/temporal_hierarchy_test_temporal_demo_daily_D_W_Q.log.diff
	 tail -10  logs/temporal_hierarchy_test_temporal_demo_daily_D_W_Q.log.diff 

temporal_hierarchy/test_temporal_demo_hourly_H_6H_12H_D.py  :  
	 -$(PYTHON)  tests/temporal_hierarchy/test_temporal_demo_hourly_H_6H_12H_D.py  >  logs/temporal_hierarchy_test_temporal_demo_hourly_H_6H_12H_D.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/temporal_hierarchy/test_temporal_demo_hourly_H_6H_12H_D.log logs/temporal_hierarchy_test_temporal_demo_hourly_H_6H_12H_D.log  >  logs/temporal_hierarchy_test_temporal_demo_hourly_H_6H_12H_D.log.diff
	 tail -10  logs/temporal_hierarchy_test_temporal_demo_hourly_H_6H_12H_D.log.diff 

temporal_hierarchy/test_temporal_demo_hourly_H_6H_12H_D_W.py  :  
	 -$(PYTHON)  tests/temporal_hierarchy/test_temporal_demo_hourly_H_6H_12H_D_W.py  >  logs/temporal_hierarchy_test_temporal_demo_hourly_H_6H_12H_D_W.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/temporal_hierarchy/test_temporal_demo_hourly_H_6H_12H_D_W.log logs/temporal_hierarchy_test_temporal_demo_hourly_H_6H_12H_D_W.log  >  logs/temporal_hierarchy_test_temporal_demo_hourly_H_6H_12H_D_W.log.diff
	 tail -10  logs/temporal_hierarchy_test_temporal_demo_hourly_H_6H_12H_D_W.log.diff 

temporal_hierarchy/test_temporal_demo_hourly_H_D.py  :  
	 -$(PYTHON)  tests/temporal_hierarchy/test_temporal_demo_hourly_H_D.py  >  logs/temporal_hierarchy_test_temporal_demo_hourly_H_D.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/temporal_hierarchy/test_temporal_demo_hourly_H_D.log logs/temporal_hierarchy_test_temporal_demo_hourly_H_D.log  >  logs/temporal_hierarchy_test_temporal_demo_hourly_H_D.log.diff
	 tail -10  logs/temporal_hierarchy_test_temporal_demo_hourly_H_D.log.diff 

temporal_hierarchy/test_temporal_demo_minutely_T_10T_30T_H.py  :  
	 -$(PYTHON)  tests/temporal_hierarchy/test_temporal_demo_minutely_T_10T_30T_H.py  >  logs/temporal_hierarchy_test_temporal_demo_minutely_T_10T_30T_H.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/temporal_hierarchy/test_temporal_demo_minutely_T_10T_30T_H.log logs/temporal_hierarchy_test_temporal_demo_minutely_T_10T_30T_H.log  >  logs/temporal_hierarchy_test_temporal_demo_minutely_T_10T_30T_H.log.diff
	 tail -10  logs/temporal_hierarchy_test_temporal_demo_minutely_T_10T_30T_H.log.diff 

temporal_hierarchy/test_temporal_demo_minutely_T_H.py  :  
	 -$(PYTHON)  tests/temporal_hierarchy/test_temporal_demo_minutely_T_H.py  >  logs/temporal_hierarchy_test_temporal_demo_minutely_T_H.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/temporal_hierarchy/test_temporal_demo_minutely_T_H.log logs/temporal_hierarchy_test_temporal_demo_minutely_T_H.log  >  logs/temporal_hierarchy_test_temporal_demo_minutely_T_H.log.diff
	 tail -10  logs/temporal_hierarchy_test_temporal_demo_minutely_T_H.log.diff 

temporal_hierarchy/test_temporal_demo_minutely_T_H_12H_D.py  :  
	 -$(PYTHON)  tests/temporal_hierarchy/test_temporal_demo_minutely_T_H_12H_D.py  >  logs/temporal_hierarchy_test_temporal_demo_minutely_T_H_12H_D.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/temporal_hierarchy/test_temporal_demo_minutely_T_H_12H_D.log logs/temporal_hierarchy_test_temporal_demo_minutely_T_H_12H_D.log  >  logs/temporal_hierarchy_test_temporal_demo_minutely_T_H_12H_D.log.diff
	 tail -10  logs/temporal_hierarchy_test_temporal_demo_minutely_T_H_12H_D.log.diff 

temporal_hierarchy/test_temporal_demo_monthly_M_2M_6M.py  :  
	 -$(PYTHON)  tests/temporal_hierarchy/test_temporal_demo_monthly_M_2M_6M.py  >  logs/temporal_hierarchy_test_temporal_demo_monthly_M_2M_6M.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/temporal_hierarchy/test_temporal_demo_monthly_M_2M_6M.log logs/temporal_hierarchy_test_temporal_demo_monthly_M_2M_6M.log  >  logs/temporal_hierarchy_test_temporal_demo_monthly_M_2M_6M.log.diff
	 tail -10  logs/temporal_hierarchy_test_temporal_demo_monthly_M_2M_6M.log.diff 

temporal_hierarchy/test_temporal_demo_monthly_M_2M_6M_12M.py  :  
	 -$(PYTHON)  tests/temporal_hierarchy/test_temporal_demo_monthly_M_2M_6M_12M.py  >  logs/temporal_hierarchy_test_temporal_demo_monthly_M_2M_6M_12M.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/temporal_hierarchy/test_temporal_demo_monthly_M_2M_6M_12M.log logs/temporal_hierarchy_test_temporal_demo_monthly_M_2M_6M_12M.log  >  logs/temporal_hierarchy_test_temporal_demo_monthly_M_2M_6M_12M.log.diff
	 tail -10  logs/temporal_hierarchy_test_temporal_demo_monthly_M_2M_6M_12M.log.diff 

temporal_hierarchy/test_temporal_demo_monthly_M_Q_A.py  :  
	 -$(PYTHON)  tests/temporal_hierarchy/test_temporal_demo_monthly_M_Q_A.py  >  logs/temporal_hierarchy_test_temporal_demo_monthly_M_Q_A.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/temporal_hierarchy/test_temporal_demo_monthly_M_Q_A.log logs/temporal_hierarchy_test_temporal_demo_monthly_M_Q_A.log  >  logs/temporal_hierarchy_test_temporal_demo_monthly_M_Q_A.log.diff
	 tail -10  logs/temporal_hierarchy_test_temporal_demo_monthly_M_Q_A.log.diff 

temporal_hierarchy/test_temporal_demo_weekly_W_2W_M_Q.py  :  
	 -$(PYTHON)  tests/temporal_hierarchy/test_temporal_demo_weekly_W_2W_M_Q.py  >  logs/temporal_hierarchy_test_temporal_demo_weekly_W_2W_M_Q.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/temporal_hierarchy/test_temporal_demo_weekly_W_2W_M_Q.log logs/temporal_hierarchy_test_temporal_demo_weekly_W_2W_M_Q.log  >  logs/temporal_hierarchy_test_temporal_demo_weekly_W_2W_M_Q.log.diff
	 tail -10  logs/temporal_hierarchy_test_temporal_demo_weekly_W_2W_M_Q.log.diff 

temporal_hierarchy/test_temporal_demo_weekly_W_Q_A.py  :  
	 -$(PYTHON)  tests/temporal_hierarchy/test_temporal_demo_weekly_W_Q_A.py  >  logs/temporal_hierarchy_test_temporal_demo_weekly_W_Q_A.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/temporal_hierarchy/test_temporal_demo_weekly_W_Q_A.log logs/temporal_hierarchy_test_temporal_demo_weekly_W_Q_A.log  >  logs/temporal_hierarchy_test_temporal_demo_weekly_W_Q_A.log.diff
	 tail -10  logs/temporal_hierarchy_test_temporal_demo_weekly_W_Q_A.log.diff 



 temporal_hierarchy :  temporal_hierarchy/test_temporal_demo_weekly_W_Q_A.py temporal_hierarchy/test_temporal_demo_weekly_W_2W_M_Q.py temporal_hierarchy/test_temporal_demo_monthly_M_Q_A.py temporal_hierarchy/test_temporal_demo_monthly_M_2M_6M_12M.py temporal_hierarchy/test_temporal_demo_monthly_M_2M_6M.py temporal_hierarchy/test_temporal_demo_minutely_T_H_12H_D.py temporal_hierarchy/test_temporal_demo_minutely_T_H.py temporal_hierarchy/test_temporal_demo_minutely_T_10T_30T_H.py temporal_hierarchy/test_temporal_demo_hourly_H_D.py temporal_hierarchy/test_temporal_demo_hourly_H_6H_12H_D_W.py temporal_hierarchy/test_temporal_demo_hourly_H_6H_12H_D.py temporal_hierarchy/test_temporal_demo_daily_D_W_Q.py temporal_hierarchy/test_temporal_demo_daily_D_W_M_Q.py temporal_hierarchy/test_temporal_demo_daily_D_W_M.py temporal_hierarchy/test_temporal_demo_daily_D_W_2W_Q.py temporal_hierarchy/test_temporal_demo_daily_D_W_2W.py temporal_hierarchy/test_temporal_demo_1.py  
 

time_res/test_ozone_Daily.py  :  
	 -$(PYTHON)  tests/time_res/test_ozone_Daily.py  >  logs/time_res_test_ozone_Daily.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/time_res/test_ozone_Daily.log logs/time_res_test_ozone_Daily.log  >  logs/time_res_test_ozone_Daily.log.diff
	 tail -10  logs/time_res_test_ozone_Daily.log.diff 

time_res/test_ozone_Hourly.py  :  
	 -$(PYTHON)  tests/time_res/test_ozone_Hourly.py  >  logs/time_res_test_ozone_Hourly.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/time_res/test_ozone_Hourly.log logs/time_res_test_ozone_Hourly.log  >  logs/time_res_test_ozone_Hourly.log.diff
	 tail -10  logs/time_res_test_ozone_Hourly.log.diff 

time_res/test_ozone_Minutely.py  :  
	 -$(PYTHON)  tests/time_res/test_ozone_Minutely.py  >  logs/time_res_test_ozone_Minutely.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/time_res/test_ozone_Minutely.log logs/time_res_test_ozone_Minutely.log  >  logs/time_res_test_ozone_Minutely.log.diff
	 tail -10  logs/time_res_test_ozone_Minutely.log.diff 

time_res/test_ozone_Secondly.py  :  
	 -$(PYTHON)  tests/time_res/test_ozone_Secondly.py  >  logs/time_res_test_ozone_Secondly.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/time_res/test_ozone_Secondly.log logs/time_res_test_ozone_Secondly.log  >  logs/time_res_test_ozone_Secondly.log.diff
	 tail -10  logs/time_res_test_ozone_Secondly.log.diff 

time_res/test_ozone_Weekly.py  :  
	 -$(PYTHON)  tests/time_res/test_ozone_Weekly.py  >  logs/time_res_test_ozone_Weekly.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/time_res/test_ozone_Weekly.log logs/time_res_test_ozone_Weekly.log  >  logs/time_res_test_ozone_Weekly.log.diff
	 tail -10  logs/time_res_test_ozone_Weekly.log.diff 



 time_res :  time_res/test_ozone_Weekly.py time_res/test_ozone_Secondly.py time_res/test_ozone_Minutely.py time_res/test_ozone_Hourly.py time_res/test_ozone_Daily.py  
 

transformations/test_ozone_transf_anscombe.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_anscombe.py  >  logs/transformations_test_ozone_transf_anscombe.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/transformations/test_ozone_transf_anscombe.log logs/transformations_test_ozone_transf_anscombe.log  >  logs/transformations_test_ozone_transf_anscombe.log.diff
	 tail -10  logs/transformations_test_ozone_transf_anscombe.log.diff 

transformations/test_ozone_transf_boxcox.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_boxcox.py  >  logs/transformations_test_ozone_transf_boxcox.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/transformations/test_ozone_transf_boxcox.log logs/transformations_test_ozone_transf_boxcox.log  >  logs/transformations_test_ozone_transf_boxcox.log.diff
	 tail -10  logs/transformations_test_ozone_transf_boxcox.log.diff 

transformations/test_ozone_transf_cumsum.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_cumsum.py  >  logs/transformations_test_ozone_transf_cumsum.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/transformations/test_ozone_transf_cumsum.log logs/transformations_test_ozone_transf_cumsum.log  >  logs/transformations_test_ozone_transf_cumsum.log.diff
	 tail -10  logs/transformations_test_ozone_transf_cumsum.log.diff 

transformations/test_ozone_transf_difference.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_difference.py  >  logs/transformations_test_ozone_transf_difference.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/transformations/test_ozone_transf_difference.log logs/transformations_test_ozone_transf_difference.log  >  logs/transformations_test_ozone_transf_difference.log.diff
	 tail -10  logs/transformations_test_ozone_transf_difference.log.diff 

transformations/test_ozone_transf_fisher.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_fisher.py  >  logs/transformations_test_ozone_transf_fisher.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/transformations/test_ozone_transf_fisher.log logs/transformations_test_ozone_transf_fisher.log  >  logs/transformations_test_ozone_transf_fisher.log.diff
	 tail -10  logs/transformations_test_ozone_transf_fisher.log.diff 

transformations/test_ozone_transf_generic.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_generic.py  >  logs/transformations_test_ozone_transf_generic.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/transformations/test_ozone_transf_generic.log logs/transformations_test_ozone_transf_generic.log  >  logs/transformations_test_ozone_transf_generic.log.diff
	 tail -10  logs/transformations_test_ozone_transf_generic.log.diff 

transformations/test_ozone_transf_logit.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_logit.py  >  logs/transformations_test_ozone_transf_logit.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/transformations/test_ozone_transf_logit.log logs/transformations_test_ozone_transf_logit.log  >  logs/transformations_test_ozone_transf_logit.log.diff
	 tail -10  logs/transformations_test_ozone_transf_logit.log.diff 

transformations/test_ozone_transf_none.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_none.py  >  logs/transformations_test_ozone_transf_none.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/transformations/test_ozone_transf_none.log logs/transformations_test_ozone_transf_none.log  >  logs/transformations_test_ozone_transf_none.log.diff
	 tail -10  logs/transformations_test_ozone_transf_none.log.diff 

transformations/test_ozone_transf_quantization.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_quantization.py  >  logs/transformations_test_ozone_transf_quantization.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/transformations/test_ozone_transf_quantization.log logs/transformations_test_ozone_transf_quantization.log  >  logs/transformations_test_ozone_transf_quantization.log.diff
	 tail -10  logs/transformations_test_ozone_transf_quantization.log.diff 

transformations/test_ozone_transf_relative_difference.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_relative_difference.py  >  logs/transformations_test_ozone_transf_relative_difference.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/transformations/test_ozone_transf_relative_difference.log logs/transformations_test_ozone_transf_relative_difference.log  >  logs/transformations_test_ozone_transf_relative_difference.log.diff
	 tail -10  logs/transformations_test_ozone_transf_relative_difference.log.diff 

transformations/test_ozone_transf_relative_difference_1.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_relative_difference_1.py  >  logs/transformations_test_ozone_transf_relative_difference_1.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/transformations/test_ozone_transf_relative_difference_1.log logs/transformations_test_ozone_transf_relative_difference_1.log  >  logs/transformations_test_ozone_transf_relative_difference_1.log.diff
	 tail -10  logs/transformations_test_ozone_transf_relative_difference_1.log.diff 



 transformations :  transformations/test_ozone_transf_relative_difference_1.py transformations/test_ozone_transf_relative_difference.py transformations/test_ozone_transf_quantization.py transformations/test_ozone_transf_none.py transformations/test_ozone_transf_logit.py transformations/test_ozone_transf_generic.py transformations/test_ozone_transf_fisher.py transformations/test_ozone_transf_difference.py transformations/test_ozone_transf_cumsum.py transformations/test_ozone_transf_boxcox.py transformations/test_ozone_transf_anscombe.py  
 

xeon-phi-parallel/test_ozone_too_many_threads.py  :  
	 -$(PYTHON)  tests/xeon-phi-parallel/test_ozone_too_many_threads.py  >  logs/xeon-phi-parallel_test_ozone_too_many_threads.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/xeon-phi-parallel/test_ozone_too_many_threads.log logs/xeon-phi-parallel_test_ozone_too_many_threads.log  >  logs/xeon-phi-parallel_test_ozone_too_many_threads.log.diff
	 tail -10  logs/xeon-phi-parallel_test_ozone_too_many_threads.log.diff 



 xeon-phi-parallel :  xeon-phi-parallel/test_ozone_too_many_threads.py  
 

xgb/test_air_passengers_xgb.py  :  
	 -$(PYTHON)  tests/xgb/test_air_passengers_xgb.py  >  logs/xgb_test_air_passengers_xgb.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/xgb/test_air_passengers_xgb.log logs/xgb_test_air_passengers_xgb.log  >  logs/xgb_test_air_passengers_xgb.log.diff
	 tail -10  logs/xgb_test_air_passengers_xgb.log.diff 

xgb/test_air_passengers_xgb_only.py  :  
	 -$(PYTHON)  tests/xgb/test_air_passengers_xgb_only.py  >  logs/xgb_test_air_passengers_xgb_only.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/xgb/test_air_passengers_xgb_only.log logs/xgb_test_air_passengers_xgb_only.log  >  logs/xgb_test_air_passengers_xgb_only.log.diff
	 tail -10  logs/xgb_test_air_passengers_xgb_only.log.diff 

xgb/test_air_passengers_xgb_only_with_custom_options.py  :  
	 -$(PYTHON)  tests/xgb/test_air_passengers_xgb_only_with_custom_options.py  >  logs/xgb_test_air_passengers_xgb_only_with_custom_options.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/xgb/test_air_passengers_xgb_only_with_custom_options.log logs/xgb_test_air_passengers_xgb_only_with_custom_options.log  >  logs/xgb_test_air_passengers_xgb_only_with_custom_options.log.diff
	 tail -10  logs/xgb_test_air_passengers_xgb_only_with_custom_options.log.diff 

xgb/test_ozone_xgb.py  :  
	 -$(PYTHON)  tests/xgb/test_ozone_xgb.py  >  logs/xgb_test_ozone_xgb.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/xgb/test_ozone_xgb.log logs/xgb_test_ozone_xgb.log  >  logs/xgb_test_ozone_xgb.log.diff
	 tail -10  logs/xgb_test_ozone_xgb.log.diff 

xgb/test_ozone_xgb_exogenous.py  :  
	 -$(PYTHON)  tests/xgb/test_ozone_xgb_exogenous.py  >  logs/xgb_test_ozone_xgb_exogenous.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/xgb/test_ozone_xgb_exogenous.log logs/xgb_test_ozone_xgb_exogenous.log  >  logs/xgb_test_ozone_xgb_exogenous.log.diff
	 tail -10  logs/xgb_test_ozone_xgb_exogenous.log.diff 

xgb/test_ozone_xgb_only.py  :  
	 -$(PYTHON)  tests/xgb/test_ozone_xgb_only.py  >  logs/xgb_test_ozone_xgb_only.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/xgb/test_ozone_xgb_only.log logs/xgb_test_ozone_xgb_only.log  >  logs/xgb_test_ozone_xgb_only.log.diff
	 tail -10  logs/xgb_test_ozone_xgb_only.log.diff 

xgb/test_ozone_xgbx_exogenous.py  :  
	 -$(PYTHON)  tests/xgb/test_ozone_xgbx_exogenous.py  >  logs/xgb_test_ozone_xgbx_exogenous.log  2>&1
	 $(PYTHON) scripts/num_diff.py  tests/references/xgb/test_ozone_xgbx_exogenous.log logs/xgb_test_ozone_xgbx_exogenous.log  >  logs/xgb_test_ozone_xgbx_exogenous.log.diff
	 tail -10  logs/xgb_test_ozone_xgbx_exogenous.log.diff 



 xgb :  xgb/test_ozone_xgbx_exogenous.py xgb/test_ozone_xgb_only.py xgb/test_ozone_xgb_exogenous.py xgb/test_ozone_xgb.py xgb/test_air_passengers_xgb_only_with_custom_options.py xgb/test_air_passengers_xgb_only.py xgb/test_air_passengers_xgb.py  
 


# ********************************************** 

all:  artificial basic_checks bugs cross_validation croston exog expsmooth func HeartRateTimeSeries heroku hierarchical  HourOfWeek model_control perf svr transformations  neuralnet real-life  time_res perfs demos xgb xeon-phi-parallel sampling temporal_hierarchy WeekOfMonth missing_data  probabilistic_forecasting lgbm perf_MedAE perf_LnQ plots multiplicative_seasonal pytorch perf_MASE_RMSSE individual_components 
	

build-test :  demos basic_checks cross_validation croston exog heroku hierarchical model_control perfs svr transformations func real-life  time_res xgb sampling HourOfWeek WeekOfMonth missing_data lgbm perf_MedAE perf_LnQ multiplicative_seasonal perf_MASE_RMSSE bugs probabilistic_forecasting plots individual_components 
	

