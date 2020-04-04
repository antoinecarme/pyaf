import pyaf.Bench.TS_datasets as tsds
import tests.artificial.process_artificial_dataset as art




art.process_dataset(N = 32 , FREQ = 'D', seed = 0, trendtype = "PolyTrend", cycle_length = 7, transform = "RelativeDifference", sigma = 0.0, exog_count = 100, ar_order = 12);