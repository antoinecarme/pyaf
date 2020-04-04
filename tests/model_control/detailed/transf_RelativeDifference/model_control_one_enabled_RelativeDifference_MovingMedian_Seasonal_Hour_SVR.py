import tests.model_control.test_ozone_custom_models_enabled as testmod


testmod.build_model( ['RelativeDifference'] , ['MovingMedian'] , ['Seasonal_Hour'] , ['SVR'] );