

RELEASE 5.0 (released on 2023-07-14)

1. Python 3.11 Support :
	Python 3.11 support #227
2. New Hardware Support :
	RISC-V Hardware Platform Validation #208
3. New Performance Measures :
	Outlier-resistant forecasting Performance Measures #209,
	Add Differentiable Variant of SMAPE Performance Measure #221
4. Model Selection Improvement :
	Investigate Model Esthetics for PyAF #212,
	Investigate Large Horizon Models #213 ,
	Revisit Model Complexity Definition #223,
	Use MASE by default for PyAF Model Selection #229
5. Signal Transformation Improvements :
	Use MaxAbsScaler for some Multiplicative Signal Transformations #235,
	Pyaf 5.0 Final Touch 8 : Use an Optimal Choice Rule for the Quantization Signal transform #239
6. Generic Modeling :
	PyAF 5.0 Final Touch 1 : discard some non-significant components #230,
	PyAF 5.0 Final Touch 2: Disable alpha in ridge regressions #231,
	Pyaf 5.0 Final Touch 5 : Add more info about Exogenous Data Used in ARX Models #236,
	Pyaf 5.0 Final Touch 7 : Improve the Guess of Window Length for Moving Average Trends #238
7. Plotting Functions Improvements and Bug Fixes :
	Bad plot for shaded area around prediction intervals in hourly data #216,
	Forecast Quantiles Plots Improved  #225,
	Pyaf 5.0 Final Touch 3 : report plot filenames in the logs #232
8. New Docs :
	Provide some UML docs for PyAF integrators #233
9.  Bug Fixes :
	Failure to build a multiplicative ozone model with Lag1 trend #220
10. PyAF "Forecast Tasks" :
	Use PyTorch as the reference deep learning framework/architecture for future projects #211,
	Automate Prototyping Activities - R-based Models #217
11. Recurrent Tasks :
	Re-run the Benchmarking process for PyAF 5.0 #222,
	Run some Sanity Checks for PyAF 5.0 #224,
	Pyaf 5.0 Final Touch 4 : Add More Tests #234,
	Pyaf 5.0 Final Touch 6 : Disable Timing Loggers by default #237
	
-------------------

RELEASE 4.0 (released on 2022-07-14)

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

