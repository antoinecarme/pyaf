import pandas as pd
import numpy as np
import pyaf.Bench.TS_datasets as tsds
import pyaf.Bench.YahooStocks as ys



symbol_lists = tsds.get_stock_web_link()
y_keys = sorted(symbol_lists.keys()) 
print(y_keys)
for k in y_keys:
    tester = ys.cYahoo_Tester(tsds.load_yahoo_stock_prices(k) , "YAHOO_STOCKS_" + k);
    tester.run_multiprocessed();
