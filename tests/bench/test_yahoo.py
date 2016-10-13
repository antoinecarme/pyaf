import pandas as pd
import numpy as np

import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds

import pyaf.CodeGen.TS_CodeGenerator as tscodegen
import pyaf.Bench.YahooStocks as ys

tester7 = ys.cYahoo_Tester(tsds.load_yahoo_stock_prices("my_test") , "YAHOO_my_test");
tester7.testAllSignals(12);
# tester7.run_multiprocessed(18);
