import pandas as pd
import numpy as np

import ForecastEngine as autof
import Bench.TS_datasets as tsds

import CodeGen.TS_CodeGenerator as tscodegen
import Bench.YahooStocks as ys

tester7 = ys.cYahoo_Tester(tsds.load_yahoo_stock_prices("my_test") , "YAHOO_my_test");
tester7.testAllSignals(12);
# tester7.run_multiprocessed(18);
