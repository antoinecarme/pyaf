import pyaf.Bench.TS_datasets as tsds
import pyaf.Bench.YahooStocks as ys
import warnings

symbol_lists = tsds.get_yahoo_symbol_lists();
y_keys = sorted(symbol_lists.keys()) 
print(y_keys)
k =  "nysecomp"
tester = ys.cYahoo_Tester(tsds.load_yahoo_stock_prices(k) , "YAHOO_STOCKS_" + k);
with warnings.catch_warnings():
    warnings.simplefilter("error")
    tester.testSignals('VRS')
