import Bench.TS_datasets as tsds
import tests.artificial.process_artificial_dataset as art




dataset = tsds.generate_random_TS(N = 32 , FREQ = 'D', seed = 0, trendtype = "constant", cycle_length = 5, transform = "pow3", sigma = 0.0, exog_count = 0);

art.process_dataset(dataset);