import os
os.environ["THEANO_FLAGS"] = "device=cpu"
os.environ["KERAS_BACKEND"] = "theano"

import tests.neuralnet.test_air_passengers_GPU as air

air.buildModel();
