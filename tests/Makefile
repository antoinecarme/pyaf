PYTHON=python3


artificial/process_artificial_dataset.py  :  
	 -$(PYTHON)  tests/artificial/process_artificial_dataset.py  >  logs/artificial_process_artificial_dataset.log  2>&1
	 -diff  logs/artificial_process_artificial_dataset.log tests/references/artificial_process_artificial_dataset.log  >  logs/artificial_process_artificial_dataset.log.diff
	 cat  logs/artificial_process_artificial_dataset.log.diff 

artificial/__init__.py  :  
	 -$(PYTHON)  tests/artificial/__init__.py  >  logs/artificial___init__.log  2>&1
	 -diff  logs/artificial___init__.log tests/references/artificial___init__.log  >  logs/artificial___init__.log.diff
	 cat  logs/artificial___init__.log.diff 



 artificial :  artificial/__init__.py artificial/process_artificial_dataset.py  
 

bugs/test_random_exogenous.py  :  
	 -$(PYTHON)  tests/bugs/test_random_exogenous.py  >  logs/bugs_test_random_exogenous.log  2>&1
	 -diff  logs/bugs_test_random_exogenous.log tests/references/bugs_test_random_exogenous.log  >  logs/bugs_test_random_exogenous.log.diff
	 cat  logs/bugs_test_random_exogenous.log.diff 

bugs/test_artificial_bug_1.py  :  
	 -$(PYTHON)  tests/bugs/test_artificial_bug_1.py  >  logs/bugs_test_artificial_bug_1.log  2>&1
	 -diff  logs/bugs_test_artificial_bug_1.log tests/references/bugs_test_artificial_bug_1.log  >  logs/bugs_test_artificial_bug_1.log.diff
	 cat  logs/bugs_test_artificial_bug_1.log.diff 

bugs/test_artificial_bug_2.py  :  
	 -$(PYTHON)  tests/bugs/test_artificial_bug_2.py  >  logs/bugs_test_artificial_bug_2.log  2>&1
	 -diff  logs/bugs_test_artificial_bug_2.log tests/references/bugs_test_artificial_bug_2.log  >  logs/bugs_test_artificial_bug_2.log.diff
	 cat  logs/bugs_test_artificial_bug_2.log.diff 



 bugs :  bugs/test_artificial_bug_2.py bugs/test_artificial_bug_1.py bugs/test_random_exogenous.py  
 

exog/test_random_exogenous.py  :  
	 -$(PYTHON)  tests/exog/test_random_exogenous.py  >  logs/exog_test_random_exogenous.log  2>&1
	 -diff  logs/exog_test_random_exogenous.log tests/references/exog_test_random_exogenous.log  >  logs/exog_test_random_exogenous.log.diff
	 cat  logs/exog_test_random_exogenous.log.diff 

exog/test_ozone_exogenous_profile.py  :  
	 -$(PYTHON)  tests/exog/test_ozone_exogenous_profile.py  >  logs/exog_test_ozone_exogenous_profile.log  2>&1
	 -diff  logs/exog_test_ozone_exogenous_profile.log tests/references/exog_test_ozone_exogenous_profile.log  >  logs/exog_test_ozone_exogenous_profile.log.diff
	 cat  logs/exog_test_ozone_exogenous_profile.log.diff 

exog/test_ozone_exogenous.py  :  
	 -$(PYTHON)  tests/exog/test_ozone_exogenous.py  >  logs/exog_test_ozone_exogenous.log  2>&1
	 -diff  logs/exog_test_ozone_exogenous.log tests/references/exog_test_ozone_exogenous.log  >  logs/exog_test_ozone_exogenous.log.diff
	 cat  logs/exog_test_ozone_exogenous.log.diff 



 exog :  exog/test_ozone_exogenous.py exog/test_ozone_exogenous_profile.py exog/test_random_exogenous.py  
 

heroku/test_air_passengers_heroku.py  :  
	 -$(PYTHON)  tests/heroku/test_air_passengers_heroku.py  >  logs/heroku_test_air_passengers_heroku.log  2>&1
	 -diff  logs/heroku_test_air_passengers_heroku.log tests/references/heroku_test_air_passengers_heroku.log  >  logs/heroku_test_air_passengers_heroku.log.diff
	 cat  logs/heroku_test_air_passengers_heroku.log.diff 

heroku/test_ozone_exog_heroku.py  :  
	 -$(PYTHON)  tests/heroku/test_ozone_exog_heroku.py  >  logs/heroku_test_ozone_exog_heroku.log  2>&1
	 -diff  logs/heroku_test_ozone_exog_heroku.log tests/references/heroku_test_ozone_exog_heroku.log  >  logs/heroku_test_ozone_exog_heroku.log.diff
	 cat  logs/heroku_test_ozone_exog_heroku.log.diff 

heroku/test_ozone_heroku.py  :  
	 -$(PYTHON)  tests/heroku/test_ozone_heroku.py  >  logs/heroku_test_ozone_heroku.log  2>&1
	 -diff  logs/heroku_test_ozone_heroku.log tests/references/heroku_test_ozone_heroku.log  >  logs/heroku_test_ozone_heroku.log.diff
	 cat  logs/heroku_test_ozone_heroku.log.diff 

heroku/test_seasonal_week_of_year.py  :  
	 -$(PYTHON)  tests/heroku/test_seasonal_week_of_year.py  >  logs/heroku_test_seasonal_week_of_year.log  2>&1
	 -diff  logs/heroku_test_seasonal_week_of_year.log tests/references/heroku_test_seasonal_week_of_year.log  >  logs/heroku_test_seasonal_week_of_year.log.diff
	 cat  logs/heroku_test_seasonal_week_of_year.log.diff 

heroku/build_generic_heroku_model.py  :  
	 -$(PYTHON)  tests/heroku/build_generic_heroku_model.py  >  logs/heroku_build_generic_heroku_model.log  2>&1
	 -diff  logs/heroku_build_generic_heroku_model.log tests/references/heroku_build_generic_heroku_model.log  >  logs/heroku_build_generic_heroku_model.log.diff
	 cat  logs/heroku_build_generic_heroku_model.log.diff 



 heroku :  heroku/build_generic_heroku_model.py heroku/test_seasonal_week_of_year.py heroku/test_ozone_heroku.py heroku/test_ozone_exog_heroku.py heroku/test_air_passengers_heroku.py  
 

hierarchical/test_grouped_signals_BU.py  :  
	 -$(PYTHON)  tests/hierarchical/test_grouped_signals_BU.py  >  logs/hierarchical_test_grouped_signals_BU.log  2>&1
	 -diff  logs/hierarchical_test_grouped_signals_BU.log tests/references/hierarchical_test_grouped_signals_BU.log  >  logs/hierarchical_test_grouped_signals_BU.log.diff
	 cat  logs/hierarchical_test_grouped_signals_BU.log.diff 

hierarchical/test_grouped_signals_TD.py  :  
	 -$(PYTHON)  tests/hierarchical/test_grouped_signals_TD.py  >  logs/hierarchical_test_grouped_signals_TD.log  2>&1
	 -diff  logs/hierarchical_test_grouped_signals_TD.log tests/references/hierarchical_test_grouped_signals_TD.log  >  logs/hierarchical_test_grouped_signals_TD.log.diff
	 cat  logs/hierarchical_test_grouped_signals_TD.log.diff 

hierarchical/test_hierarchy_AU.py  :  
	 -$(PYTHON)  tests/hierarchical/test_hierarchy_AU.py  >  logs/hierarchical_test_hierarchy_AU.log  2>&1
	 -diff  logs/hierarchical_test_hierarchy_AU.log tests/references/hierarchical_test_hierarchy_AU.log  >  logs/hierarchical_test_hierarchy_AU.log.diff
	 cat  logs/hierarchical_test_hierarchy_AU.log.diff 

hierarchical/test_hierarchy_AU_BU.py  :  
	 -$(PYTHON)  tests/hierarchical/test_hierarchy_AU_BU.py  >  logs/hierarchical_test_hierarchy_AU_BU.log  2>&1
	 -diff  logs/hierarchical_test_hierarchy_AU_BU.log tests/references/hierarchical_test_hierarchy_AU_BU.log  >  logs/hierarchical_test_hierarchy_AU_BU.log.diff
	 cat  logs/hierarchical_test_hierarchy_AU_BU.log.diff 

hierarchical/test_grouped_signals_AllMethods.py  :  
	 -$(PYTHON)  tests/hierarchical/test_grouped_signals_AllMethods.py  >  logs/hierarchical_test_grouped_signals_AllMethods.log  2>&1
	 -diff  logs/hierarchical_test_grouped_signals_AllMethods.log tests/references/hierarchical_test_grouped_signals_AllMethods.log  >  logs/hierarchical_test_grouped_signals_AllMethods.log.diff
	 cat  logs/hierarchical_test_grouped_signals_AllMethods.log.diff 

hierarchical/test_hierarchy_AU_TD.py  :  
	 -$(PYTHON)  tests/hierarchical/test_hierarchy_AU_TD.py  >  logs/hierarchical_test_hierarchy_AU_TD.log  2>&1
	 -diff  logs/hierarchical_test_hierarchy_AU_TD.log tests/references/hierarchical_test_hierarchy_AU_TD.log  >  logs/hierarchical_test_hierarchy_AU_TD.log.diff
	 cat  logs/hierarchical_test_hierarchy_AU_TD.log.diff 

hierarchical/test_hierarchy_AU_OC.py  :  
	 -$(PYTHON)  tests/hierarchical/test_hierarchy_AU_OC.py  >  logs/hierarchical_test_hierarchy_AU_OC.log  2>&1
	 -diff  logs/hierarchical_test_hierarchy_AU_OC.log tests/references/hierarchical_test_hierarchy_AU_OC.log  >  logs/hierarchical_test_hierarchy_AU_OC.log.diff
	 cat  logs/hierarchical_test_hierarchy_AU_OC.log.diff 

hierarchical/test_grouped_signals_OC.py  :  
	 -$(PYTHON)  tests/hierarchical/test_grouped_signals_OC.py  >  logs/hierarchical_test_grouped_signals_OC.log  2>&1
	 -diff  logs/hierarchical_test_grouped_signals_OC.log tests/references/hierarchical_test_grouped_signals_OC.log  >  logs/hierarchical_test_grouped_signals_OC.log.diff
	 cat  logs/hierarchical_test_grouped_signals_OC.log.diff 

hierarchical/test_hierarchy_AU_MO.py  :  
	 -$(PYTHON)  tests/hierarchical/test_hierarchy_AU_MO.py  >  logs/hierarchical_test_hierarchy_AU_MO.log  2>&1
	 -diff  logs/hierarchical_test_hierarchy_AU_MO.log tests/references/hierarchical_test_hierarchy_AU_MO.log  >  logs/hierarchical_test_hierarchy_AU_MO.log.diff
	 cat  logs/hierarchical_test_hierarchy_AU_MO.log.diff 

hierarchical/test_grouped_signals_MO.py  :  
	 -$(PYTHON)  tests/hierarchical/test_grouped_signals_MO.py  >  logs/hierarchical_test_grouped_signals_MO.log  2>&1
	 -diff  logs/hierarchical_test_grouped_signals_MO.log tests/references/hierarchical_test_grouped_signals_MO.log  >  logs/hierarchical_test_grouped_signals_MO.log.diff
	 cat  logs/hierarchical_test_grouped_signals_MO.log.diff 

hierarchical/test_grouped.py  :  
	 -$(PYTHON)  tests/hierarchical/test_grouped.py  >  logs/hierarchical_test_grouped.log  2>&1
	 -diff  logs/hierarchical_test_grouped.log tests/references/hierarchical_test_grouped.log  >  logs/hierarchical_test_grouped.log.diff
	 cat  logs/hierarchical_test_grouped.log.diff 

hierarchical/test_hierarchy_AU_AllMethods.py  :  
	 -$(PYTHON)  tests/hierarchical/test_hierarchy_AU_AllMethods.py  >  logs/hierarchical_test_hierarchy_AU_AllMethods.log  2>&1
	 -diff  logs/hierarchical_test_hierarchy_AU_AllMethods.log tests/references/hierarchical_test_hierarchy_AU_AllMethods.log  >  logs/hierarchical_test_hierarchy_AU_AllMethods.log.diff
	 cat  logs/hierarchical_test_hierarchy_AU_AllMethods.log.diff 



 hierarchical :  hierarchical/test_hierarchy_AU_AllMethods.py hierarchical/test_grouped.py hierarchical/test_grouped_signals_MO.py hierarchical/test_hierarchy_AU_MO.py hierarchical/test_grouped_signals_OC.py hierarchical/test_hierarchy_AU_OC.py hierarchical/test_hierarchy_AU_TD.py hierarchical/test_grouped_signals_AllMethods.py hierarchical/test_hierarchy_AU_BU.py hierarchical/test_hierarchy_AU.py hierarchical/test_grouped_signals_TD.py hierarchical/test_grouped_signals_BU.py  
 

model_control/test_ozone_all_models_enabled.py  :  
	 -$(PYTHON)  tests/model_control/test_ozone_all_models_enabled.py  >  logs/model_control_test_ozone_all_models_enabled.log  2>&1
	 -diff  logs/model_control_test_ozone_all_models_enabled.log tests/references/model_control_test_ozone_all_models_enabled.log  >  logs/model_control_test_ozone_all_models_enabled.log.diff
	 cat  logs/model_control_test_ozone_all_models_enabled.log.diff 

model_control/test_ozone_custom_models_enabled.py  :  
	 -$(PYTHON)  tests/model_control/test_ozone_custom_models_enabled.py  >  logs/model_control_test_ozone_custom_models_enabled.log  2>&1
	 -diff  logs/model_control_test_ozone_custom_models_enabled.log tests/references/model_control_test_ozone_custom_models_enabled.log  >  logs/model_control_test_ozone_custom_models_enabled.log.diff
	 cat  logs/model_control_test_ozone_custom_models_enabled.log.diff 

model_control/test_ozone_no_models_enabled.py  :  
	 -$(PYTHON)  tests/model_control/test_ozone_no_models_enabled.py  >  logs/model_control_test_ozone_no_models_enabled.log  2>&1
	 -diff  logs/model_control_test_ozone_no_models_enabled.log tests/references/model_control_test_ozone_no_models_enabled.log  >  logs/model_control_test_ozone_no_models_enabled.log.diff
	 cat  logs/model_control_test_ozone_no_models_enabled.log.diff 



 model_control :  model_control/test_ozone_no_models_enabled.py model_control/test_ozone_custom_models_enabled.py model_control/test_ozone_all_models_enabled.py  
 

perf/test_perf1.py  :  
	 -$(PYTHON)  tests/perf/test_perf1.py  >  logs/perf_test_perf1.log  2>&1
	 -diff  logs/perf_test_perf1.log tests/references/perf_test_perf1.log  >  logs/perf_test_perf1.log.diff
	 cat  logs/perf_test_perf1.log.diff 

perf/test_cycles_full_long_long.py  :  
	 -$(PYTHON)  tests/perf/test_cycles_full_long_long.py  >  logs/perf_test_cycles_full_long_long.log  2>&1
	 -diff  logs/perf_test_cycles_full_long_long.log tests/references/perf_test_cycles_full_long_long.log  >  logs/perf_test_cycles_full_long_long.log.diff
	 cat  logs/perf_test_cycles_full_long_long.log.diff 

perf/test_parallel.py  :  
	 -$(PYTHON)  tests/perf/test_parallel.py  >  logs/perf_test_parallel.log  2>&1
	 -diff  logs/perf_test_parallel.log tests/references/perf_test_parallel.log  >  logs/perf_test_parallel.log.diff
	 cat  logs/perf_test_parallel.log.diff 

perf/test_perf2.py  :  
	 -$(PYTHON)  tests/perf/test_perf2.py  >  logs/perf_test_perf2.log  2>&1
	 -diff  logs/perf_test_perf2.log tests/references/perf_test_perf2.log  >  logs/perf_test_perf2.log.diff
	 cat  logs/perf_test_perf2.log.diff 

perf/test_ozone_long_series_2.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_long_series_2.py  >  logs/perf_test_ozone_long_series_2.log  2>&1
	 -diff  logs/perf_test_ozone_long_series_2.log tests/references/perf_test_ozone_long_series_2.log  >  logs/perf_test_ozone_long_series_2.log.diff
	 cat  logs/perf_test_ozone_long_series_2.log.diff 

perf/test_ozone_long_series.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_long_series.py  >  logs/perf_test_ozone_long_series.log  2>&1
	 -diff  logs/perf_test_ozone_long_series.log tests/references/perf_test_ozone_long_series.log  >  logs/perf_test_ozone_long_series.log.diff
	 cat  logs/perf_test_ozone_long_series.log.diff 

perf/test_cycles_full_long.py  :  
	 -$(PYTHON)  tests/perf/test_cycles_full_long.py  >  logs/perf_test_cycles_full_long.log  2>&1
	 -diff  logs/perf_test_cycles_full_long.log tests/references/perf_test_cycles_full_long.log  >  logs/perf_test_cycles_full_long.log.diff
	 cat  logs/perf_test_cycles_full_long.log.diff 

perf/test_ozone_debug_perf.py  :  
	 -$(PYTHON)  tests/perf/test_ozone_debug_perf.py  >  logs/perf_test_ozone_debug_perf.log  2>&1
	 -diff  logs/perf_test_ozone_debug_perf.log tests/references/perf_test_ozone_debug_perf.log  >  logs/perf_test_ozone_debug_perf.log.diff
	 cat  logs/perf_test_ozone_debug_perf.log.diff 



 perf :  perf/test_ozone_debug_perf.py perf/test_cycles_full_long.py perf/test_ozone_long_series.py perf/test_ozone_long_series_2.py perf/test_perf2.py perf/test_parallel.py perf/test_cycles_full_long_long.py perf/test_perf1.py  
 

svr/test_ozone_svr_only.py  :  
	 -$(PYTHON)  tests/svr/test_ozone_svr_only.py  >  logs/svr_test_ozone_svr_only.log  2>&1
	 -diff  logs/svr_test_ozone_svr_only.log tests/references/svr_test_ozone_svr_only.log  >  logs/svr_test_ozone_svr_only.log.diff
	 cat  logs/svr_test_ozone_svr_only.log.diff 

svr/test_air_passengers_svr.py  :  
	 -$(PYTHON)  tests/svr/test_air_passengers_svr.py  >  logs/svr_test_air_passengers_svr.log  2>&1
	 -diff  logs/svr_test_air_passengers_svr.log tests/references/svr_test_air_passengers_svr.log  >  logs/svr_test_air_passengers_svr.log.diff
	 cat  logs/svr_test_air_passengers_svr.log.diff 

svr/test_ozone_svr.py  :  
	 -$(PYTHON)  tests/svr/test_ozone_svr.py  >  logs/svr_test_ozone_svr.log  2>&1
	 -diff  logs/svr_test_ozone_svr.log tests/references/svr_test_ozone_svr.log  >  logs/svr_test_ozone_svr.log.diff
	 cat  logs/svr_test_ozone_svr.log.diff 

svr/test_air_passengers_svr_only.py  :  
	 -$(PYTHON)  tests/svr/test_air_passengers_svr_only.py  >  logs/svr_test_air_passengers_svr_only.log  2>&1
	 -diff  logs/svr_test_air_passengers_svr_only.log tests/references/svr_test_air_passengers_svr_only.log  >  logs/svr_test_air_passengers_svr_only.log.diff
	 cat  logs/svr_test_air_passengers_svr_only.log.diff 



 svr :  svr/test_air_passengers_svr_only.py svr/test_ozone_svr.py svr/test_air_passengers_svr.py svr/test_ozone_svr_only.py  
 

transformations/test_ozone_transf_relative_difference.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_relative_difference.py  >  logs/transformations_test_ozone_transf_relative_difference.log  2>&1
	 -diff  logs/transformations_test_ozone_transf_relative_difference.log tests/references/transformations_test_ozone_transf_relative_difference.log  >  logs/transformations_test_ozone_transf_relative_difference.log.diff
	 cat  logs/transformations_test_ozone_transf_relative_difference.log.diff 

transformations/test_ozone_transf_anscombe.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_anscombe.py  >  logs/transformations_test_ozone_transf_anscombe.log  2>&1
	 -diff  logs/transformations_test_ozone_transf_anscombe.log tests/references/transformations_test_ozone_transf_anscombe.log  >  logs/transformations_test_ozone_transf_anscombe.log.diff
	 cat  logs/transformations_test_ozone_transf_anscombe.log.diff 

transformations/test_ozone_transf_none.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_none.py  >  logs/transformations_test_ozone_transf_none.log  2>&1
	 -diff  logs/transformations_test_ozone_transf_none.log tests/references/transformations_test_ozone_transf_none.log  >  logs/transformations_test_ozone_transf_none.log.diff
	 cat  logs/transformations_test_ozone_transf_none.log.diff 

transformations/test_ozone_transf_fisher.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_fisher.py  >  logs/transformations_test_ozone_transf_fisher.log  2>&1
	 -diff  logs/transformations_test_ozone_transf_fisher.log tests/references/transformations_test_ozone_transf_fisher.log  >  logs/transformations_test_ozone_transf_fisher.log.diff
	 cat  logs/transformations_test_ozone_transf_fisher.log.diff 

transformations/test_ozone_transf_difference.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_difference.py  >  logs/transformations_test_ozone_transf_difference.log  2>&1
	 -diff  logs/transformations_test_ozone_transf_difference.log tests/references/transformations_test_ozone_transf_difference.log  >  logs/transformations_test_ozone_transf_difference.log.diff
	 cat  logs/transformations_test_ozone_transf_difference.log.diff 

transformations/test_ozone_transf_logit.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_logit.py  >  logs/transformations_test_ozone_transf_logit.log  2>&1
	 -diff  logs/transformations_test_ozone_transf_logit.log tests/references/transformations_test_ozone_transf_logit.log  >  logs/transformations_test_ozone_transf_logit.log.diff
	 cat  logs/transformations_test_ozone_transf_logit.log.diff 

transformations/test_ozone_transf_boxcox.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_boxcox.py  >  logs/transformations_test_ozone_transf_boxcox.log  2>&1
	 -diff  logs/transformations_test_ozone_transf_boxcox.log tests/references/transformations_test_ozone_transf_boxcox.log  >  logs/transformations_test_ozone_transf_boxcox.log.diff
	 cat  logs/transformations_test_ozone_transf_boxcox.log.diff 

transformations/test_ozone_transf_generic.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_generic.py  >  logs/transformations_test_ozone_transf_generic.log  2>&1
	 -diff  logs/transformations_test_ozone_transf_generic.log tests/references/transformations_test_ozone_transf_generic.log  >  logs/transformations_test_ozone_transf_generic.log.diff
	 cat  logs/transformations_test_ozone_transf_generic.log.diff 

transformations/test_ozone_transf_cumsum.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_cumsum.py  >  logs/transformations_test_ozone_transf_cumsum.log  2>&1
	 -diff  logs/transformations_test_ozone_transf_cumsum.log tests/references/transformations_test_ozone_transf_cumsum.log  >  logs/transformations_test_ozone_transf_cumsum.log.diff
	 cat  logs/transformations_test_ozone_transf_cumsum.log.diff 

transformations/test_ozone_transf_quantization.py  :  
	 -$(PYTHON)  tests/transformations/test_ozone_transf_quantization.py  >  logs/transformations_test_ozone_transf_quantization.log  2>&1
	 -diff  logs/transformations_test_ozone_transf_quantization.log tests/references/transformations_test_ozone_transf_quantization.log  >  logs/transformations_test_ozone_transf_quantization.log.diff
	 cat  logs/transformations_test_ozone_transf_quantization.log.diff 



 transformations :  transformations/test_ozone_transf_quantization.py transformations/test_ozone_transf_cumsum.py transformations/test_ozone_transf_generic.py transformations/test_ozone_transf_boxcox.py transformations/test_ozone_transf_logit.py transformations/test_ozone_transf_difference.py transformations/test_ozone_transf_fisher.py transformations/test_ozone_transf_none.py transformations/test_ozone_transf_anscombe.py transformations/test_ozone_transf_relative_difference.py  
 

bench/test_yahoo.py  :  
	 -$(PYTHON)  tests/bench/test_yahoo.py  >  logs/bench_test_yahoo.log  2>&1
	 -diff  logs/bench_test_yahoo.log tests/references/bench_test_yahoo.log  >  logs/bench_test_yahoo.log.diff
	 cat  logs/bench_test_yahoo.log.diff 

bench/test_M4_C0808_failure.py  :  
	 -$(PYTHON)  tests/bench/test_M4_C0808_failure.py  >  logs/bench_test_M4_C0808_failure.log  2>&1
	 -diff  logs/bench_test_M4_C0808_failure.log tests/references/bench_test_M4_C0808_failure.log  >  logs/bench_test_M4_C0808_failure.log.diff
	 cat  logs/bench_test_M4_C0808_failure.log.diff 

bench/test_M2Comp.py  :  
	 -$(PYTHON)  tests/bench/test_M2Comp.py  >  logs/bench_test_M2Comp.log  2>&1
	 -diff  logs/bench_test_M2Comp.log tests/references/bench_test_M2Comp.log  >  logs/bench_test_M2Comp.log.diff
	 cat  logs/bench_test_M2Comp.log.diff 

bench/test_MWH_datasets.py  :  
	 -$(PYTHON)  tests/bench/test_MWH_datasets.py  >  logs/bench_test_MWH_datasets.log  2>&1
	 -diff  logs/bench_test_MWH_datasets.log tests/references/bench_test_MWH_datasets.log  >  logs/bench_test_MWH_datasets.log.diff
	 cat  logs/bench_test_MWH_datasets.log.diff 

bench/test_stock_prices.py  :  
	 -$(PYTHON)  tests/bench/test_stock_prices.py  >  logs/bench_test_stock_prices.log  2>&1
	 -diff  logs/bench_test_stock_prices.log tests/references/bench_test_stock_prices.log  >  logs/bench_test_stock_prices.log.diff
	 cat  logs/bench_test_stock_prices.log.diff 

bench/test_yahoo_cac40.py  :  
	 -$(PYTHON)  tests/bench/test_yahoo_cac40.py  >  logs/bench_test_yahoo_cac40.log  2>&1
	 -diff  logs/bench_test_yahoo_cac40.log tests/references/bench_test_yahoo_cac40.log  >  logs/bench_test_yahoo_cac40.log.diff
	 cat  logs/bench_test_yahoo_cac40.log.diff 

bench/test_M4Comp.py  :  
	 -$(PYTHON)  tests/bench/test_M4Comp.py  >  logs/bench_test_M4Comp.log  2>&1
	 -diff  logs/bench_test_M4Comp.log tests/references/bench_test_M4Comp.log  >  logs/bench_test_M4Comp.log.diff
	 cat  logs/bench_test_M4Comp.log.diff 

bench/test_M3Comp.py  :  
	 -$(PYTHON)  tests/bench/test_M3Comp.py  >  logs/bench_test_M3Comp.log  2>&1
	 -diff  logs/bench_test_M3Comp.log tests/references/bench_test_M3Comp.log  >  logs/bench_test_M3Comp.log.diff
	 cat  logs/bench_test_M3Comp.log.diff 

bench/test_M1Comp.py  :  
	 -$(PYTHON)  tests/bench/test_M1Comp.py  >  logs/bench_test_M1Comp.log  2>&1
	 -diff  logs/bench_test_M1Comp.log tests/references/bench_test_M1Comp.log  >  logs/bench_test_M1Comp.log.diff
	 cat  logs/bench_test_M1Comp.log.diff 

bench/test_MComp.py  :  
	 -$(PYTHON)  tests/bench/test_MComp.py  >  logs/bench_test_MComp.log  2>&1
	 -diff  logs/bench_test_MComp.log tests/references/bench_test_MComp.log  >  logs/bench_test_MComp.log.diff
	 cat  logs/bench_test_MComp.log.diff 

bench/test_NN5.py  :  
	 -$(PYTHON)  tests/bench/test_NN5.py  >  logs/bench_test_NN5.log  2>&1
	 -diff  logs/bench_test_NN5.log tests/references/bench_test_NN5.log  >  logs/bench_test_NN5.log.diff
	 cat  logs/bench_test_NN5.log.diff 

bench/test_NN3.py  :  
	 -$(PYTHON)  tests/bench/test_NN3.py  >  logs/bench_test_NN3.log  2>&1
	 -diff  logs/bench_test_NN3.log tests/references/bench_test_NN3.log  >  logs/bench_test_NN3.log.diff
	 cat  logs/bench_test_NN3.log.diff 

bench/test_artificial_one_dataset.py  :  
	 -$(PYTHON)  tests/bench/test_artificial_one_dataset.py  >  logs/bench_test_artificial_one_dataset.log  2>&1
	 -diff  logs/bench_test_artificial_one_dataset.log tests/references/bench_test_artificial_one_dataset.log  >  logs/bench_test_artificial_one_dataset.log.diff
	 cat  logs/bench_test_artificial_one_dataset.log.diff 



 bench :  bench/test_artificial_one_dataset.py bench/test_NN3.py bench/test_NN5.py bench/test_MComp.py bench/test_M1Comp.py bench/test_M3Comp.py bench/test_M4Comp.py bench/test_yahoo_cac40.py bench/test_stock_prices.py bench/test_MWH_datasets.py bench/test_M2Comp.py bench/test_M4_C0808_failure.py bench/test_yahoo.py  
 

func/test_ozone_with_logging.py  :  
	 -$(PYTHON)  tests/func/test_ozone_with_logging.py  >  logs/func_test_ozone_with_logging.log  2>&1
	 -diff  logs/func_test_ozone_with_logging.log tests/references/func_test_ozone_with_logging.log  >  logs/func_test_ozone_with_logging.log.diff
	 cat  logs/func_test_ozone_with_logging.log.diff 

func/test_cycle.py  :  
	 -$(PYTHON)  tests/func/test_cycle.py  >  logs/func_test_cycle.log  2>&1
	 -diff  logs/func_test_cycle.log tests/references/func_test_cycle.log  >  logs/func_test_cycle.log.diff
	 cat  logs/func_test_cycle.log.diff 

func/test_ozone.py  :  
	 -$(PYTHON)  tests/func/test_ozone.py  >  logs/func_test_ozone.log  2>&1
	 -diff  logs/func_test_ozone.log tests/references/func_test_ozone.log  >  logs/func_test_ozone.log.diff
	 cat  logs/func_test_ozone.log.diff 

func/test_cycles_full.py  :  
	 -$(PYTHON)  tests/func/test_cycles_full.py  >  logs/func_test_cycles_full.log  2>&1
	 -diff  logs/func_test_cycles_full.log tests/references/func_test_cycles_full.log  >  logs/func_test_cycles_full.log.diff
	 cat  logs/func_test_cycles_full.log.diff 

func/test_air_passengers.py  :  
	 -$(PYTHON)  tests/func/test_air_passengers.py  >  logs/func_test_air_passengers.log  2>&1
	 -diff  logs/func_test_air_passengers.log tests/references/func_test_air_passengers.log  >  logs/func_test_air_passengers.log.diff
	 cat  logs/func_test_air_passengers.log.diff 

func/test_const_signal.py  :  
	 -$(PYTHON)  tests/func/test_const_signal.py  >  logs/func_test_const_signal.log  2>&1
	 -diff  logs/func_test_const_signal.log tests/references/func_test_const_signal.log  >  logs/func_test_const_signal.log.diff
	 cat  logs/func_test_const_signal.log.diff 



 func :  func/test_const_signal.py func/test_air_passengers.py func/test_cycles_full.py func/test_ozone.py func/test_cycle.py func/test_ozone_with_logging.py  
 

neuralnet/test_ozone_tensorflow.py  :  
	 -$(PYTHON)  tests/neuralnet/test_ozone_tensorflow.py  >  logs/neuralnet_test_ozone_tensorflow.log  2>&1
	 -diff  logs/neuralnet_test_ozone_tensorflow.log tests/references/neuralnet_test_ozone_tensorflow.log  >  logs/neuralnet_test_ozone_tensorflow.log.diff
	 cat  logs/neuralnet_test_ozone_tensorflow.log.diff 

neuralnet/test_air_passengers_GPU.py  :  
	 -$(PYTHON)  tests/neuralnet/test_air_passengers_GPU.py  >  logs/neuralnet_test_air_passengers_GPU.log  2>&1
	 -diff  logs/neuralnet_test_air_passengers_GPU.log tests/references/neuralnet_test_air_passengers_GPU.log  >  logs/neuralnet_test_air_passengers_GPU.log.diff
	 cat  logs/neuralnet_test_air_passengers_GPU.log.diff 

neuralnet/test_ozone_rnn_only.py  :  
	 -$(PYTHON)  tests/neuralnet/test_ozone_rnn_only.py  >  logs/neuralnet_test_ozone_rnn_only.log  2>&1
	 -diff  logs/neuralnet_test_ozone_rnn_only.log tests/references/neuralnet_test_ozone_rnn_only.log  >  logs/neuralnet_test_ozone_rnn_only.log.diff
	 cat  logs/neuralnet_test_ozone_rnn_only.log.diff 

neuralnet/test_air_passengers_tensorflow.py  :  
	 -$(PYTHON)  tests/neuralnet/test_air_passengers_tensorflow.py  >  logs/neuralnet_test_air_passengers_tensorflow.log  2>&1
	 -diff  logs/neuralnet_test_air_passengers_tensorflow.log tests/references/neuralnet_test_air_passengers_tensorflow.log  >  logs/neuralnet_test_air_passengers_tensorflow.log.diff
	 cat  logs/neuralnet_test_air_passengers_tensorflow.log.diff 

neuralnet/test_ozone__GPU.py  :  
	 -$(PYTHON)  tests/neuralnet/test_ozone__GPU.py  >  logs/neuralnet_test_ozone__GPU.log  2>&1
	 -diff  logs/neuralnet_test_ozone__GPU.log tests/references/neuralnet_test_ozone__GPU.log  >  logs/neuralnet_test_ozone__GPU.log.diff
	 cat  logs/neuralnet_test_ozone__GPU.log.diff 

neuralnet/test_air_passengers_rnn_only.py  :  
	 -$(PYTHON)  tests/neuralnet/test_air_passengers_rnn_only.py  >  logs/neuralnet_test_air_passengers_rnn_only.log  2>&1
	 -diff  logs/neuralnet_test_air_passengers_rnn_only.log tests/references/neuralnet_test_air_passengers_rnn_only.log  >  logs/neuralnet_test_air_passengers_rnn_only.log.diff
	 cat  logs/neuralnet_test_air_passengers_rnn_only.log.diff 



 neuralnet :  neuralnet/test_air_passengers_rnn_only.py neuralnet/test_ozone__GPU.py neuralnet/test_air_passengers_tensorflow.py neuralnet/test_ozone_rnn_only.py neuralnet/test_air_passengers_GPU.py neuralnet/test_ozone_tensorflow.py  
 

real-life/test_sof_example2.py  :  
	 -$(PYTHON)  tests/real-life/test_sof_example2.py  >  logs/real-life_test_sof_example2.log  2>&1
	 -diff  logs/real-life_test_sof_example2.log tests/references/real-life_test_sof_example2.log  >  logs/real-life_test_sof_example2.log.diff
	 cat  logs/real-life_test_sof_example2.log.diff 

real-life/test_sof_example.py  :  
	 -$(PYTHON)  tests/real-life/test_sof_example.py  >  logs/real-life_test_sof_example.log  2>&1
	 -diff  logs/real-life_test_sof_example.log tests/references/real-life_test_sof_example.log  >  logs/real-life_test_sof_example.log.diff
	 cat  logs/real-life_test_sof_example.log.diff 



 real-life :  real-life/test_sof_example.py real-life/test_sof_example2.py  
 

time_res/test_ozone_Daily.py  :  
	 -$(PYTHON)  tests/time_res/test_ozone_Daily.py  >  logs/time_res_test_ozone_Daily.log  2>&1
	 -diff  logs/time_res_test_ozone_Daily.log tests/references/time_res_test_ozone_Daily.log  >  logs/time_res_test_ozone_Daily.log.diff
	 cat  logs/time_res_test_ozone_Daily.log.diff 

time_res/test_ozone_Weekly.py  :  
	 -$(PYTHON)  tests/time_res/test_ozone_Weekly.py  >  logs/time_res_test_ozone_Weekly.log  2>&1
	 -diff  logs/time_res_test_ozone_Weekly.log tests/references/time_res_test_ozone_Weekly.log  >  logs/time_res_test_ozone_Weekly.log.diff
	 cat  logs/time_res_test_ozone_Weekly.log.diff 

time_res/test_ozone_Minutely.py  :  
	 -$(PYTHON)  tests/time_res/test_ozone_Minutely.py  >  logs/time_res_test_ozone_Minutely.log  2>&1
	 -diff  logs/time_res_test_ozone_Minutely.log tests/references/time_res_test_ozone_Minutely.log  >  logs/time_res_test_ozone_Minutely.log.diff
	 cat  logs/time_res_test_ozone_Minutely.log.diff 

time_res/test_ozone_Secondly.py  :  
	 -$(PYTHON)  tests/time_res/test_ozone_Secondly.py  >  logs/time_res_test_ozone_Secondly.log  2>&1
	 -diff  logs/time_res_test_ozone_Secondly.log tests/references/time_res_test_ozone_Secondly.log  >  logs/time_res_test_ozone_Secondly.log.diff
	 cat  logs/time_res_test_ozone_Secondly.log.diff 

time_res/test_ozone_Hourly.py  :  
	 -$(PYTHON)  tests/time_res/test_ozone_Hourly.py  >  logs/time_res_test_ozone_Hourly.log  2>&1
	 -diff  logs/time_res_test_ozone_Hourly.log tests/references/time_res_test_ozone_Hourly.log  >  logs/time_res_test_ozone_Hourly.log.diff
	 cat  logs/time_res_test_ozone_Hourly.log.diff 



 time_res :  time_res/test_ozone_Hourly.py time_res/test_ozone_Secondly.py time_res/test_ozone_Minutely.py time_res/test_ozone_Weekly.py time_res/test_ozone_Daily.py  
 


# ********************************************** 

all:  artificial bugs exog heroku hierarchical model_control perf svr transformations bench func neuralnet real-life  time_res 
	

build-test :  bugs exog heroku hierarchical model_control svr transformations func real-life  time_res 
	

