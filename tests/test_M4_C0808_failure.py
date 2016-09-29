import pandas as pd
import numpy as np
import AutoForecast as autof


trainfile = "data/bugs/M4_C0808_failure.csv";
#cols = ["ID" , "time", "AirPassengers"];
df = pd.read_csv(trainfile, sep=r',', engine='python');
    
lAutoF = autof.cForecastEngine()
lAutoF

print(df.head());
print(df.tail());
H = 2
lAutoF.train(df , "Date" , "C0808", H);
lAutoF.getModelInfo();

lAutoF.mSignalDecomposition.mBestTransformation.mTimeInfo.mResolution

lAutoF.standrdPlots("my_ozone");

dfapp_in = df.copy();
dfapp_in.tail()

#H = 12
dfapp_out = lAutoF.forecast(dfapp_in, H);
dfapp_out.tail(2 * H)
print("Forecast Columns " , dfapp_out.columns);
Forecast_DF = dfapp_out[["Date" , "C0808", "C0808" + '_BestModelForecast']]
print(Forecast_DF.info())
print("Forecasts\n" , Forecast_DF.tail(H).values);

print("\n\n<ModelInfo>")
print(lAutoF.to_json());
print("</ModelInfo>\n\n")
print("\n\n<Forecast>")
print(Forecast_DF.to_json(date_format='iso'))
print("</Forecast>\n\n")

