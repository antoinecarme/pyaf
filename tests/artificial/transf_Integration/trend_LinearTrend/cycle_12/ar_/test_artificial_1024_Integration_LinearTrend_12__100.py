import pyaf.Bench.TS_datasets as tsds
import tests.artificial.process_artificial_dataset as art




art.process_dataset(N = 1024 , FREQ = 'D', seed = 0, trendtype = "LinearTrend", cycle_length = 12, transform = "Integration", sigma = 0.0, exog_count = 100, ar_order = 0);