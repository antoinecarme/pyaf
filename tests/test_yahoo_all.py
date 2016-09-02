import pandas as pd
import numpy as np
import Bench.TS_datasets as tsds
import Bench.YahooStocks as ys

symbol_lists = tsds.get_yahoo_symbol_lists();
for k in symbol_lists.keys():
    tester = ys.cYahoo_Tester(tsds.load_yahoo_stock_prices(k) , "YAHOO_STOCKS_" + k);
    tester.run_multiprocessed(18);
