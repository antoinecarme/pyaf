import pyaf.Bench.TS_datasets as tsds
import pyaf.tests.artificial.process_artificial_dataset as art




dataset = tsds.generate_random_TS(N = 32 , FREQ = 'D', seed = 0, trendtype = "linear", cycle_length = 12, transform = "diff", sigma = 0.0, exog_count = 20, ar_order = 12);

art.process_dataset(dataset);