import pandas as pd
import numpy as np

import AutoForecast.ForecastEngine as autof
import AutoForecast.Bench.TS_datasets as tsds

import AutoForecast.CodeGen.TS_CodeGenerator as tscodegen

import sys, traceback

#get_ipython().magic('matplotlib inline')


def test_MWH_datset(name):
    print("AUTO_FORECAST_MWH_TEST_START" , name)
    try:
        b1 = tsds.load_mwh_dataset(name);
    except:
        print("unknown MWH dataset" , name)
        traceback.print_exc(file=sys.stdout)
        print("AUTO_FORECAST_MWH_TEST_FAILED" , name)
        return;

    df = b1.mPastData
    df.info();
    print(df.head(12));
    print(df.tail(12));
    
    # df.tail(10)
    # df[:-10].tail()
    # df[:-10:-1]
    # df.describe()


    lDecomp = SigDec.cSignalDecomposition()
    lDecomp

    H = b1.mHorizon;
    try:
        lDecomp.train(df , b1.mTimeVar , b1.mSignalVar, H);
    except util.ForecastError as error:
        print('caught this error: ' + repr(error))
        print("AUTO_FORECAST_MWH_TEST_FAILED" , name)
        return;
    except:
        print("Training Failed" , name)
        traceback.print_exc(file=sys.stdout)
        print(repr(traceback.extract_stack()))
        #print(repr(traceback.format_stack()))
        #traceback.print_tb(err.__traceback__)
        print("AUTO_FORECAST_MWH_TEST_FAILED" , name)
        return;
    lDecomp.getModelInfo();

    lDecomp.mBestTransformation.mTimeInfo.mResolution
    
    dfapp_in = df.copy();
    dfapp_in.tail()
    
    # H = 12
    dfapp_out = lDecomp.forecast(dfapp_in, H);
    dfapp_out.tail(2 * H)
    print("Forecast Columns " , dfapp_out.columns);
    Forecast_DF = dfapp_out[[b1.mTimeVar , b1.mSignalVar, b1.mSignalVar + '_BestModelForecast']]
    print(Forecast_DF.info())
    print("Forecasts\n" , Forecast_DF.tail(H).values);
    
    print("\n\n<ModelInfo>")
    print(lDecomp.to_json());
    print("</ModelInfo>\n\n")
    print("\n\n<Forecast>")
    print(Forecast_DF.to_json(date_format='iso'))
    print("</Forecast>\n\n")

    print("AUTO_FORECAST_MWH_TEST_SUCCEEDED" , name)


datasets = "10-6 11-2 9-10 9-11 9-12 9-13 9-17a 9-17b 9-1 9-2 9-3 9-4 9-5 9-9 advert adv_sale airline bankdata beer2 bicoal books boston bricksq canadian capital cars cement computer condmilk cow cpi_mel deaths dexter dj dole dowjones eknives elco elec2 elec elecnew ex2_6 ex5_2 expendit fancy french fsales gas housing hsales2 hsales huron ibm2 input invent15 jcars kkong labour lynx milk mink mortal motel motion olympic ozone paris pcv petrol pigs plastics pollutn prodc pulppric qsales res running sales schizo shampoo sheep ship shipex strikes temperat ukdeaths ustreas wagesuk wn wnoise writing".split(" ");

#datasets = "milk".split(" ");

for ds in datasets:
    test_MWH_datset(ds);
