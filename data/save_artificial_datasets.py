import pandas as pd
import numpy as np
import pyaf.Bench.TS_datasets as tsds
import pyaf.Bench.Artificial as art
import warnings


with warnings.catch_warnings():
    warnings.simplefilter("error")
    tsds.load_artificial_datsets("S")
    tsds.load_artificial_datsets("M")
    tsds.load_artificial_datsets("L")
    tsds.load_artificial_datsets("XL")
