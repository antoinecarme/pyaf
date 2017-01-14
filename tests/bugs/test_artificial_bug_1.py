import pyaf.Bench.TS_datasets as tsds
import tests.artificial.process_artificial_dataset as art
import warnings


with warnings.catch_warnings():
    warnings.simplefilter("error")

    dataset = tsds.generate_random_TS(N = 200 , FREQ = 'D', seed = 0, trendtype = "linear", cycle_length = 48, transform = "exp", sigma = 4.0, exog_count = 0);

    art.process_dataset(dataset);
