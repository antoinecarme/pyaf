import Bench.TS_datasets as tsds
import tests.artificial.process_artificial_dataset as art




dataset = tsds.generate_random_TS(N = 1024 , FREQ = 'D', seed = 0, trendtype = "linear", cycle_length = 30, transform = "cumsum", sigma = 0.0, exog_count = 0);

art.process_dataset(dataset);