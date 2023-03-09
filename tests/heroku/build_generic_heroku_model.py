

def build_model(iDict):
    import pyaf.WS.WS_Backend as be

    lModel = be.cWSModel();
    lModel.from_dict(iDict);
    
    lEngine = lModel.mForecastEngine
    df = lModel.mTrainDataFrame.copy();
    H = lModel.mHorizon
    

    dfapp_in = df.copy();
    dfapp_in.tail()

    dfapp_out = lEngine.forecast(dfapp_in, H);
    # dfapp_out.to_csv("outputs/ozone_apply_out.csv")
    dfapp_out.tail(2 * H)
    print("Forecast Columns " , dfapp_out.columns);
    Forecast_DF = dfapp_out[[lModel.mTimeVar , lModel.mSignalVar, lModel.mSignalVar + '_Forecast']]
    print(Forecast_DF.info())
    print("Forecasts\n" , Forecast_DF.tail(H));
    
    print("\n\n<ModelInfo>")
    print(lEngine.to_json());
    print("</ModelInfo>\n\n")
    print("\n\n<Forecast>")
    print(Forecast_DF.tail(2*H).to_json(date_format='iso'))
    print("</Forecast>\n\n")

    return lModel;
