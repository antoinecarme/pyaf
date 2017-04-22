import pyaf.Bench.YahooStocks as ys
import pyaf.Bench.TS_datasets as tsds

tester7 = ys.cYahoo_Tester(tsds.load_yahoo_stock_prices("my_test") , "YAHOO_my_test");
tester7.testAllSignals(12);
# tester7.run_multiprocessed(18);
