import pyaf.Bench.TS_datasets as tsds
import tests.artificial.process_artificial_dataset as art




art.process_dataset(N = 1024 , FREQ = 'D', seed = 0, trendtype = "ConstantTrend", cycle_length = 7, transform = "Quantization", sigma = 0.0, exog_count = 0, ar_order = 0);