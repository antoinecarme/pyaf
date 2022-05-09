# TODO

## save / reload the model
self.serialize() => to_dict every where + json

## forecast future values => update datasets + forecast

## signal transformation => almost done
- cumulative (integrated) tranformation

## Cutting into Estim/Valid (NE*H , NV*H) => cross-validation
1. ratio_E = 0.5
2. N_train = N_Estim + N_Valid + N_test (=H)
3. N_Estim = ratio_E * (N_train - H) 

## prediction intervals
1. improve plotting ... shaded area around prediction intervals
   neeed prediction intervals first

## use cross validation 
2. http://robjhyndman.com/hyndsight/tscvexample/
3. book : https://www.otexts.org/fpp
4. https://www.otexts.org/fpp/2/5


## Residual diagnostics	
	https://www.otexts.org/fpp/2/6

## other trends
1. Exponential smoothing      ****************************************************
2. exogenous variables   ********************************************
3. moving average(N). OK
4. moving median(N). OK

## other cycles
	1. seasonal
	2. user holidays etc (external tables?)

## other AR
1. ARX OK.
2. VAR ?
2. order control (look at timedelta ??).

## Configuration (Options)
1.  activate/disable transfromations/models/decomposition.  
2. configure trends
3. configure cycles (CycleLength = ?)
	- cycle length should be in [5, 7, 12, 24 , 30, 60]
4. configure ARs (p = ?)
5. processing : threads etc
  

## Benchmarking
1. MComp
2. NN5
3. NN3
4. Yahoo stocks
	
## speedup things
1. python is sloooooooooooow (cython ?)
2. multiprocessing seems OK

## timedelta adaptive estimation
1. allow user control.
2. truncate timedelta to the nearest unit.
3. avoid saturday/sunday if not present in the dataset.
  
## LOGGING

## cross validation for time series
=> http://robjhyndman.com/talks/MelbourneRUG.pdf

## feature selection (remove unnecessary lags and exogenous variables).
===> smaller model => smaller SQL code !!!


## real-life examples :
http://stackoverflow.com/questions/10302261/forecasting-time-series-data

## better graphics
https://stanford.edu/~mwaskom/software/seaborn/

## GitHub Topics
autoregressive benchmark cycles data-frame exogenous forecasting heroku hierarchical horizon  jupyter machine-learning-library pandas restful-api scikit-learn seasonal sql sql-generation time-series trends 

# Forecast Competition

http://eem2017.com/program/forecast-competition

In cooperation with our technical sponsor, we will provide you with a set of different weather input factors, e.g. wind direction, with which you are to forecast the power generation of a wind power plant portfolio. You may participate individually or as a team. The data input is organised in a realistic setting. 




## 2021-08

1. Prediction Interval Quality
2. Multiplicative Decompositions (log transform ?). 2022-05-09 OK (#178 : https://github.com/antoinecarme/pyaf/issues/178)
3. PyTorch : 2022-05-09 OK (#199 : https://github.com/antoinecarme/pyaf/issues/199)

