import pyaf.ForecastEngine as autof
import pyaf.Bench.TS_datasets as tsds

b1 = tsds.load_ozone()
df = b1.mPastData



def test_decomposition_type(iDecomp):

    lEngine = autof.cForecastEngine()
    lEngine

    H = b1.mHorizon;
    lEngine.mOptions.set_active_decomposition_types([ iDecomp ]);
    lEngine.mOptions.set_active_transformations([ 'None' ]);
    lEngine.mOptions.set_active_trends([ 'LinearTrend' ]);
    lEngine.mOptions.set_active_periodics([ 'BestCycle' ]);
    lEngine.mOptions.set_active_autoregressions([ 'NoAR' ]);
    lEngine.train(df , b1.mTimeVar , b1.mSignalVar, H);
    lEngine.getModelInfo();
    
    lEngine.standardPlots("outputs/my_ozone_decomposition_type_" + iDecomp + "_");

    dfapp_in = df.copy();
    dfapp_in.tail()

    #H = 12
    dfapp_out = lEngine.forecast(dfapp_in, H);
    #dfapp_out.to_csv("outputs/ozone_apply_out.csv")
    dfapp_out.tail(2 * H)
    print("Forecast Columns " , dfapp_out.columns);
    Forecast_DF = dfapp_out[[b1.mTimeVar , b1.mSignalVar, b1.mSignalVar + '_Forecast']]
    print(Forecast_DF.info())
    print("Forecasts\n" , Forecast_DF.tail(H));
    
    print("\n\n<ModelInfo>")
    print(lEngine.to_json());
    print("</ModelInfo>\n\n")
    print("\n\n<Forecast>")
    print(Forecast_DF.tail(2*H).to_json(date_format='iso'))
    print("</Forecast>\n\n")



def test_all_decomposition_types():
    import pyaf.TS.Options as tsopts
    lKnownDecomp = tsopts.cModelControl.gKnownDecompositionTypes
    for lDecomp in lKnownDecomp:
        test_decomposition_type(lDecomp)

test_all_decomposition_types()
