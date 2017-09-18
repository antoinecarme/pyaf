
import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds




b1 = tsds.generate_random_TS(N = 40000 , FREQ = 'S', seed = 0, trendtype = "constant", cycle_length = 24, transform = "", sigma = 0.1, exog_count = 0);

df = b1.mPastData

lEngine = autof.cForecastEngine()
lEngine

H = 12;
lEngine.mOptions.mFilterSeasonals = False;
lEngine.mOptions.mParallelMode = False;
lEngine.mOptions.mDebugPerformance = True;
lEngine.train(df , b1.mTimeVar , b1.mSignalVar, H);
lEngine.getModelInfo();
print(lEngine.mSignalDecomposition.mTrPerfDetails.head());

