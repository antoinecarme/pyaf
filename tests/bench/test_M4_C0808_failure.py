import pandas as pd
import numpy as np
import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds


trainfile = "data/bugs/M4_C0808_failure.csv";
#cols = ["ID" , "time", "AirPassengers"];
df = pd.read_csv(trainfile, sep=r',', engine='python');
    
lEngine = autof.cForecastEngine()
lEngine

print(df.head());
print(df.tail());
H = 2
lEngine.train(df , "Date" , "C0808", H);
lEngine.getModelInfo();

lEngine.mSignalDecomposition.mBestModel.mTimeInfo.mResolution

lEngine.standardPlots("outputs/M4_C0808_failure");

dfapp_in = df.copy();
dfapp_in.tail()

#H = 12
dfapp_out = lEngine.forecast(dfapp_in, H);
dfapp_out.tail(2 * H)
print("Forecast Columns " , dfapp_out.columns);
Forecast_DF = dfapp_out[["Date" , "C0808", "C0808" + '_Forecast']]
print(Forecast_DF.info())
print("Forecasts\n" , Forecast_DF.tail(H).values);

print("\n\n<ModelInfo>")
print(lEngine.to_json());
print("</ModelInfo>\n\n")
print("\n\n<Forecast>")
print(Forecast_DF.tail(2*H).to_json(date_format='iso'))
print("</Forecast>\n\n")

