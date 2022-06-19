
RELEASE 4.0 (expected on 2022-07-14)

1. Python 3.10 support #186 
2. Add Multiplicative Models/Seasonals #178 
3. Speed Performance Improvements : #190 , #191 
4. Exogenous data support improvements :  #193, #197, #198 
5. PyAF support for ARM64 Architecture #187 
6. PyTorch support : #199  
7. Improved Logging : #185 
8. Bug Fixes : #156,  #179,  #182, #184
9. Release Process : Pre-release Benchmarks #194 
10. Release Process : Profiling and Warning Hunts #195 
11. Release Process : Review Existing Docs #196, #35

-----------------

RELEASE 3.0 (released on 2021-07-14)

1. Python 3.9 support #149
2. Probabilistic Forecasting  : Forecast quantiles (#140), CRPS (#74), Plots and Docs (#158).
3. Add LightGBM based models #143
4. Add more Performance Measures : MedAE (#144) , LnQ ( #43 )
5. PyAF Powerpc support (IBM S822xx) #160
6. More Parallelization Efforts (#145)
7. Add Missing Data Imputation Methods (#146 )
8. Improved long signals modeling (#167)
9. Warning Hunts (#153)
10. Some Bug Fixes (#163, #142, #168).
11. Switched to Circle-CI (#164)
12. Plot Functions Improvement #169
13. Model Complexity Improvement (#171)
14. Documentation review/corrections (#174)

---------------

RELEASE 2.0 (released on 2020-07-14)

1. Time column is normalized frequently leading to a performance issue. Profiling. Significant speedup. Issue #121
2. Corrected PyPi packaging. Issue #123
3. Allow using exogenous data in hierarchical forecasting models. Issue #124
4. Properly handle very large signals. Add Sampling. Issue #126
5. Add temporal hierarchical forecasting. Issue #127
6. Analyze Business Seasonals (HourOfWeek and derivatives) . Issue #131
7. Improved logs (More model details). Issue #133, #134, #135
8. More robust scycles (use target median instead of target mean encoding). Issue #132
9. Analyze Business Seasonals (WeekOfMonth and derivatives). Issue #137
10. Improved JSON output (added Model Options). Issue #136
11. Improved cpu usage (parallelization) for hierarchical models. Issue #115
12. Speedups in multiple places : forecasts generation, plotting,  AR Modelling (feature selection).
13. Last minute fixes

