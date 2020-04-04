import pyaf.Bench.TS_datasets as tsds
import tests.artificial.process_artificial_dataset as art




art.process_dataset(N = 128 , FREQ = 'D', seed = 0, trendtype = "MovingAverage", cycle_length = 12, transform = "Logit", sigma = 0.0, exog_count = 100, ar_order = 12);