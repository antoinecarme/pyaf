import Bench.TS_datasets as tsds
import tests.artificial.process_artificial_dataset as art




dataset = tsds.generate_random_TS(N = 128 , FREQ = 'D', seed = 0, trendtype = "linear", cycle_length = 0, transform = "exp", sigma = 0.0, exog_count = 100);

art.process_dataset(dataset);