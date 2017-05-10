
import tests.neuralnet.test_air_passengers_GPU as air
# CUDA_VISIBLE_DEVICES

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

air.buildModel(iParallel = True);
air.buildModel(iParallel = False);
