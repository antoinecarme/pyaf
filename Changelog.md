
RELEASE 2.0 (expected 2020-07-14)

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
