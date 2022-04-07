import pyaf.Bench.TS_datasets as tsds
import pyaf.Bench.YahooStocks as ys
import warnings

k =  "nysecomp"
tester = ys.cYahoo_Tester(tsds.load_yahoo_stock_prices(k, stock="VRS") , "YAHOO_STOCKS_" + k);
with warnings.catch_warnings():
    warnings.simplefilter("error")
    tester.testSignals('VRS')
