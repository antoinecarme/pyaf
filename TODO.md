# TODO

## ~~save / reload the model~~
self.serialize() => to_dict every where + json

## ~~forecast future values => update datasets + forecast~~

## ~~signal transformation~~ almost done
- ~~cumulative (integrated) tranformation~~ OK

## ~~Cutting into Estim/Valid (NE*H , NV*H)~~  => cross-validation
1. ~~ratio_E = 0.5~~
2. ~~N_train = N_Estim + N_Valid + N_test (=H)~~
3. ~~N_Estim = ratio_E * (N_train - H) ~~

## ~~prediction intervals
1. ~~improve plotting ... shaded area around prediction intervals
   neeed prediction intervals first

## ~~use cross validation 
2. http://robjhyndman.com/hyndsight/tscvexample/
3. book : https://www.otexts.org/fpp
4. https://www.otexts.org/fpp/2/5


## Residual diagnostics	
	https://www.otexts.org/fpp/2/6

## other trends
1. Exponential smoothing      
2. ~~exogenous variables~~  OK.
3. ~~moving average(N)~~. OK
4. ~~moving median(N)~~. OK

## other cycles
	1. seasonal
	2. user holidays etc (external tables?)

## other AR
1. ~~ARX~~ OK.
2. VAR ?
2. order control (look at timedelta ??).

## Configuration (Options)
1.  ~~activate/disable transfromations/models/decomposition~~.  OK
2. ~~configure trends~~ OK.
3. ~~configure cycles (CycleLength = ?)~~. OK
	- cycle length should be in [5, 7, 12, 24 , 30, 60]
4. ~~configure ARs (p = ?)~~ OK.
5. ~~processing : threads etc~~ OK.
  

## Benchmarking
1. ~~MComp~~ OK.
2. ~~NN5~~ OK.
3. ~~NN3~~ OK.
4. ~~Yahoo stocks~~ OK.
	
## speedup things
1. python is sloooooooooooow (cython ?)
2. multiprocessing seems OK

## timedelta adaptive estimation
1. ~~allow user control~~. OK.
2. ~~truncate timedelta to the nearest unit.~~ OK.
3. avoid saturday/sunday if not present in the dataset.
  
## LOGGING

## ~~cross validation for time series
=> http://robjhyndman.com/talks/MelbourneRUG.pdf

## ~~feature selection (remove unnecessary lags and exogenous variables).
===> smaller model => smaller SQL code !!!


## ~~real-life examples :
http://stackoverflow.com/questions/10302261/forecasting-time-series-data ~~~~ OK

## better graphics
https://stanford.edu/~mwaskom/software/seaborn/. Let someone else do that !!!

## ~~GitHub Topics
autoregressive benchmark cycles data-frame exogenous forecasting heroku hierarchical horizon  jupyter machine-learning-library pandas restful-api scikit-learn seasonal sql sql-generation time-series trends 

# Forecast Competition

http://eem2017.com/program/forecast-competition

In cooperation with our technical sponsor, we will provide you with a set of different weather input factors, e.g. wind direction, with which you are to forecast the power generation of a wind power plant portfolio. You may participate individually or as a team. The data input is organised in a realistic setting. 




## 2021-08

1. Prediction Interval Quality
2. ~~Multiplicative Decompositions (log transform ?)~~. 2022-05-09 OK (#178 : https://github.com/antoinecarme/pyaf/issues/178)
3. ~~PyTorch~~ : 2022-05-09 OK (#199 : https://github.com/antoinecarme/pyaf/issues/199)

## 2023-07-14

### Outliers in Time Series

1. Outliers detection. https://otexts.com/fpp2/missing-outliers.html
2. In ARX Models, category 1 : data quality / wrong input. Can be removed. https://otexts.com/fpp2/regression-evaluation.html
3. In ARX Models, category 2 : Natural, Simply different data. Should not be removed.
https://otexts.com/fpp2/regression-evaluation.html
4. X11 decomposition : The process is entirely automatic and tends to be highly robust to outliers and level shifts in the time series. https://otexts.com/fpp2/x11.html
5. STL decomposition : outliers may affect the remainder component. https://otexts.com/fpp2/stl.html
6. Referenced in https://github.com/antoinecarme/pyaf/issues/230
7. Outliers removal : estimate the trends/cycles/AR models with an estimation dataset that does not contain the outliers. Can be capped without removal ?
8. Outliers Reporting. Scatter Plots ? https://otexts.com/fpp2/scatterplots.html
9. Generic view (not only for time series): https://en.wikipedia.org/wiki/Outlier
10. Detection method : Tukey's fences, based on measures such as the interquartile range. range = [Q1 - k * (Q3 - Q1) , Q3 + k * (Q3 - Q1)], k > 0. Simple, non-parametric, robust.
11. Flags/forecast outputs : Tukey uses k = 1.5 to flag as "outlier" and k=3 to flag as "far out".
12. k value can be used as a training option to remove more-or-less outliers. Default : 3 ?
13. Interquartile range (IQR = Q3 - Q1):  middle 50%.  https://en.wikipedia.org/wiki/Interquartile_range
14. IQR is a Robust measure of scale : for N(0, sigma) , IQR = 1.349 sigma. https://en.wikipedia.org/wiki/Robust_measures_of_scale
15. Add more candidates with and without oultiers (ARX and ARXO)? Not sure. KISS.
16. tsoutliers : The tsoutliers() function in the forecast package for R is useful for identifying anomalies in a time series. Excellent !!!
17. tsoutliers : This package implements a procedure based on the approach described in Chen and Liu (1993) for automatic detection of outliers in time series. Innovational outliers, additive outliers, level shifts, temporary changes and seasonal level shifts are considered
18.  Joint Estimation of Model Parameters and Outlier Effects in Time Series. Chung Chen & Lon-Mu Liu.  Journal of the American Statistical Association Volume 88, 1993 - Issue 421
19. http://cran.r-project.org/web/packages/tsoutliers/tsoutliers.pdf
20. tsoutliers: Custom processing for outliers. Iterative MSTL + Smoothing trends + IQR + linear interpolation.  https://robjhyndman.com/hyndsight/tsoutliers
21. tsoutliers Gold price dataset (fpp2 package) : The gold price data contains daily morning gold prices in US dollars from 1 January 1985 to 31 March 1989.
22. tsoutliers hicp dataset : Harmonised indices of consumer prices in the Euro area.
23. tsoutliers ipi dataset : Industrial production indices in the manufacturing sector of European Monetary Union countries.
24.tsoutliers bde9915 dataset : seasonal outliers ? Kaiser, R., and Maravall, A. (1999). Seasonal Outliers in Time Series. Banco de España, Servicio de Estudios. Working paper number 9915.
