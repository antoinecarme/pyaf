import pandas as pd
import numpy as np
import pyaf.Bench.TS_datasets as tsds
import pyaf.Bench.YahooStocks as ys

k = "nysecomp"
tester = ys.cYahoo_Tester(tsds.load_yahoo_stock_prices(k) , "YAHOO_STOCKS_" + k);
tester.testSignals('VRS');
