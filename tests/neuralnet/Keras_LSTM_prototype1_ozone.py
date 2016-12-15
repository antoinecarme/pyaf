

import Keras_LSTM_prototype1_gen as nntest

nntest.full_test('data/ozone-la.csv', 'Ozone', 24, 100)

#for ep in [10 , 40, 160, 640]:
#    for lags in [8 ,32]:
#        full_test(lags, ep)

