import os

def createDirIfNeeded(dirname):
    try:
        os.mkdir(dirname);
    except:
        pass




lKnownTransformations = ['None', 'Difference', 'RelativeDifference',
                         'Integration', 'BoxCox',
                         'Quantization', 'Logit',
                         'Fisher', 'Anscombe'];
lKnownTrends = ['ConstantTrend', 
                'Lag1Trend', 'LinearTrend', 'PolyTrend', 
                'MovingAverage', 'MovingMedian'];
lKnownPeriodics = ['NoCycle', 'BestCycle',
                   'Seasonal_MonthOfYear' ,
                   'Seasonal_Second' ,
                   'Seasonal_Minute' ,
                   'Seasonal_Hour' ,
                   'Seasonal_DayOfWeek' ,
                   'Seasonal_DayOfMonth',
                   'Seasonal_WeekOfYear'];
lKnownAutoRegressions = ['NoAR' , 'AR' , 'ARX' , 'SVR' , 'MLP' , 'LSTM'];

createDirIfNeeded("tests/model_control/detailed/");

for transf in lKnownTransformations:
    createDirIfNeeded("tests/model_control/detailed/transf_" + transf);
    for trend in lKnownTrends:
        for per in lKnownPeriodics:
            for autoreg in lKnownAutoRegressions:
                filename= "tests/model_control/detailed/transf_" + str(transf) + "/model_control_one_enabled_" + str(transf) + "_" + str(trend) + "_" + str(per) + "_" + str(autoreg) + ".py";
                file = open(filename, "w");
                print("WRTITING_FILE" , filename);
                file.write("import tests.model_control.test_ozone_custom_models_enabled as testmod\n");
                file.write("\n\ntestmod.build_model( ['" + str(transf) + "'] , ['" + str(trend) + "'] , ['" + str(per) + "'] , ['" + str(autoreg) + "'] );");
                file.close();
