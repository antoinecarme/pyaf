import tests.model_control.test_ozone_custom_models_enabled as testmod


testmod.build_model( ['RelativeDifference'] , ['ConstantTrend'] , ['NoCycle'] , ['LSTM'] );