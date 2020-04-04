import tests.model_control.test_ozone_custom_models_enabled as testmod


testmod.build_model( ['BoxCox'] , ['MovingMedian'] , ['Seasonal_MonthOfYear'] , ['AR'] );