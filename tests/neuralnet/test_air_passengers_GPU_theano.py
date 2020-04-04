import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=0.1,exception_verbosity=high,dnn.enabled=False,optimizer=None,print_active_device=False"
os.environ["KERAS_BACKEND"] = "theano"

import tests.neuralnet.test_air_passengers_GPU as air

air.buildModel();
