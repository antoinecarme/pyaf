import numpy as np
import pandas as pd
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

import theano
theano.config.openmp = True

#get_ipython().magic('matplotlib inline')
#get_ipython().magic('env OMP_NUM_THREADS=12')




def create_dataset_lags(dataset, signal, nblags=1):
    df = dataset.copy();
    for i in range(nblags):
        name = signal + "_" + str(i+1);
        df[name] = dataset[signal].shift(i+1);
    return df

def load_dataset(source , signal):
    dataframe = pd.read_csv(source, engine='python')
    return dataframe ;

def get_lag_names(signal, nblags):
    names = [];
    for i in range(nblags):
        name = signal + "_" + str(i+1);
        names.append(name);
    return names;

def cut_dataset(dataframe , signal, lags):
    train_size = int(dataframe.shape[0] * 0.67)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(dataframe[signal][0:train_size].values.ravel())
    dataframe['scaled_' + signal] = scaler.transform(dataframe[signal].values.ravel())
    lagged_df = create_dataset_lags(dataframe, 'scaled_' + signal, lags)
    (train_df, test_df) = (lagged_df[0:train_size] , lagged_df[train_size:])

    return (scaler, train_df , test_df)


def train_model(train_df , signal, lags, epochs):
    model = Sequential()
    model.add(LSTM(lags, input_dim=lags))
    
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    
    lag_names = get_lag_names('scaled_' + signal , lags);
    N = train_df.shape[0] - lags
    NEstim = (N * 4) // 5;
    trainX = train_df[lag_names][lags:].values
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    trainY = train_df['scaled_' + signal][lags:].values
    estimX = trainX[0:NEstim]
    estimY = trainY[0:NEstim]
    valX = trainX[ NEstim : ]
    valY = trainY[ NEstim : ]

    lStopCallback = callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    lHistory = model.fit(estimX, estimY, nb_epoch=epochs, batch_size=1, validation_data=(valX , valY), verbose=2, 
                        callbacks=[lStopCallback])
    print(lHistory.__dict__)
    return model;

def plot_model(model):
    from IPython.display import SVG
    from keras.utils.visualize_util import model_to_dot

    SVG(model_to_dot(model, show_shapes=True, show_layer_names=True).create(prog='dot', format='svg'))

    

def compute_L2_MAPE(signal , estimator):
    lMean = np.mean( (signal - estimator)**2 );
    lMAPE = np.mean( np.abs((signal - estimator) / signal ));
    lL2 = np.sqrt(lMean);
    return (lL2 , lMAPE);


def predict_signal(model, scaler, signal, nblags, train_df, test_df, idataframe):
    lag_names = get_lag_names('scaled_' + signal , nblags);
    trainX = train_df[lag_names].values
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    trainPredict = model.predict(trainX)
    testX = test_df[lag_names].values
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    testPredict = model.predict(testX)
    trainPredict = scaler.inverse_transform(trainPredict[nblags:])
    testPredict = scaler.inverse_transform(testPredict)
    
    # calculate root mean squared error
    lTrainL2 = compute_L2_MAPE(train_df[signal][nblags:].values, trainPredict)
    lTestL2 = compute_L2_MAPE(test_df[signal].values, testPredict)
    print('TRAIN_TEST_RMSE_MAPE', lTrainL2 , lTestL2)

    out_df = pd.DataFrame()
    out_df = idataframe.copy();
    out_N = out_df.shape[0]
    out_df['Time'] = range(out_N)
    # out_df['scaled_output'] = 0;
    lSeries = pd.Series(np.full(nblags , np.nan));
    lSeries1 = pd.Series(trainPredict.ravel());
    lSeries2 = pd.Series(testPredict.ravel());
    # print(out_N , lSeries.shape[0], lSeries1.shape[0], lSeries2.shape[0])
    lSeries = lSeries.append(lSeries1);
    lSeries = lSeries.append(lSeries2);
    out_df['output'] = lSeries.values;
    out_df['output'] = out_df['output'];
    return out_df;


def full_test(dataset, signal, nblags , epochs):
    full_df = load_dataset(dataset , signal);
    (scaler, train_df, test_df) = cut_dataset(full_df, signal , nblags);
    model = train_model(train_df, signal , nblags, epochs);
    print(model.__dict__);
    plot_model(model);
    out_df = predict_signal(model, scaler, signal, nblags, train_df, test_df, full_df);
    lNewName = signal + "_" + str(nblags) +  "_" + str(epochs) 
    out_df[lNewName] = out_df[signal]
    out_df.plot('Time' , [lNewName,  'output'] , figsize=(22,12));

    
    

