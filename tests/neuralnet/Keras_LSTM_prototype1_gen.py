import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


import theano
theano.config.openmp = True

#%matplotlib inline
#%env OMP_NUM_THREADS=12



def create_dataset_lags(dataset, signal, nblags=1):
    df = dataset.copy();
    for i in range(nblags):
        name = signal + "_" + str(i+1);
        df[name] = dataset[signal].shift(i+1);
    return df

def load_dataset(source , signal):
    dataframe = pandas.read_csv(source, engine='python')
    print(dataframe.columns)
    return dataframe;

def get_lag_names(signal, nblags):
    names = [];
    for i in range(nblags):
        name = signal + "_" + str(i+1);
        names.append(name);
    return names;

def cut_dataset(dataframe , signal, lags):
    train_size = int(dataframe.shape[0] * 0.67)
    lagged_df = create_dataset_lags(dataframe, signal, lags)
    (train_df, test_df) = (lagged_df[0:train_size] , lagged_df[train_size:])
    return (train_df , test_df)


def train_model(train_df , signal, lags, epochs):
    model = Sequential()
    model.add(LSTM(4, input_dim=lags))
    
    model.add(Dense(1))
    model.compile(loss='mape', optimizer='adam')
    
    lag_names = get_lag_names(signal , lags);
    trainX = train_df[lag_names][lags:].values
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    trainY = train_df[signal][lags:].values

    lHistory = model.fit(trainX, trainY, nb_epoch=epochs, batch_size=1, validation_split=0.1, verbose=0)
    print(lHistory.__dict__)
    return model;

def plot_model(model):
    from IPython.display import SVG
    from keras.utils.visualize_util import model_to_dot

    SVG(model_to_dot(model, show_shapes=True, show_layer_names=True).create(prog='dot', format='svg'))

    

def compute_L2_MAPE(signal , estimator):
    lMean = numpy.mean( (signal - estimator)**2 );
    lMAPE = numpy.mean( numpy.abs((signal - estimator) / signal ));
    lL2 = numpy.sqrt(lMean);
    return (lL2 , lMAPE);


def predict_signal(model, signal, nblags, train_df, test_df, idataframe):
    lag_names = get_lag_names(signal , nblags);
    trainX = train_df[lag_names].values
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    trainPredict = model.predict(trainX)
    testX = test_df[lag_names].values
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    testPredict = model.predict(testX)
    
    # calculate root mean squared error
    lTrainL2 = compute_L2_MAPE(train_df[signal][nblags:].values, trainPredict[nblags:])
    lTestL2 = compute_L2_MAPE(test_df[signal].values, testPredict)
    print('TRAIN_TEST_RMSE_MAPE', lTrainL2 , lTestL2)

    out_df = pandas.DataFrame()
    out_df = idataframe.copy();
    out_N = out_df.shape[0]
    out_df['Time'] = range(out_N)
    # out_df['scaled_output'] = 0;
    lSeries = pandas.Series();
    lSeries[:] = numpy.nan
    lSeries1 = pandas.Series(trainPredict.ravel());
    lSeries2 = pandas.Series(testPredict.ravel());
    # print(out_N , lSeries.shape[0], lSeries1.shape[0], lSeries2.shape[0])
    lSeries = lSeries.append(lSeries1);
    lSeries = lSeries.append(lSeries2);
    out_df['output'] = lSeries.values;
    out_df['output'] = out_df['output'];
    return out_df;

def full_test(dataset, signal, nblags , epochs):
    ozone_df = load_dataset(dataset , signal);
    (train_df, test_df) = cut_dataset(ozone_df, signal , nblags);
    model = train_model(train_df, signal , nblags, epochs);
    print(model.__dict__);
    plot_model(model);
    out_df = predict_signal(model, signal, nblags, train_df, test_df, ozone_df);
    lNewName = signal + "_" + str(nblags) +  "_" + str(epochs) 
    out_df[lNewName] = out_df[signal]
    out_df.plot('Time' , [lNewName,  'output'] , figsize=(22,12));

    
    
