from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import *;
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D,Flatten
from keras.layers import LSTM, Activation
from util.ff1data import readDataSets
import numpy
from keras.optimizers import RMSprop, SGD;
import tensorflow as tf;
import numpy as np;

import matplotlib;


import matplotlib.pyplot as plt;

#use price change percentage to predict

def mean_pred(y_true, y_pred):
    correct_prediction = tf.equal( tf.cast(tf.round(y_true), tf.int32), tf.cast(tf.round(y_pred), tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def earning_pred(y_true, y_pred):
    positive_pred = tf.reduce_sum(tf.cast(tf.round(y_pred), tf.int32));
    correct_prediction = tf.equal( tf.cast(tf.round(y_true), tf.int32), tf.cast(tf.round(y_pred), tf.int32))
    positive_correct = tf.logical_and( correct_prediction, tf.greater(y_true, tf.constant(0.5, dtype=tf.float32)))
    return tf.cond( tf.greater(positive_pred, tf.constant(0)), lambda:tf.reduce_sum(tf.to_int32( positive_correct)) / positive_pred, lambda:tf.to_int32(0) );
   
     

def buildModel():        
    model = Sequential()
    model.add(Conv1D(100, 5, strides =1, activation='relu', input_shape=(100, 1)))
    print(model.output_shape)
    model.add(MaxPooling1D(5))
    print(model.output_shape)
    model.add(Conv1D(100, 3, strides = 1, activation='relu'))
    print(model.output_shape)
    model.add(MaxPooling1D(5))
    print(model.output_shape)
    model.add(Conv1D(20, 3, strides = 1, activation='relu'))
    print(model.output_shape)
    #model.add(GlobalAveragePooling1D())
    #model.add(Dropout(0.5))
    
    #model.add(Flatten())
    print(model.output_shape)
    model.add(Dense(10, activation='sigmoid', name="d1"))
    
    model.add(Dense(1, activation='sigmoid', name="d2"))
    print(model.output_shape)
    myOptimizer = SGD(lr=0.02, clipvalue=0.5);
    model.compile(loss='mse', optimizer=myOptimizer, metrics=['accuracy', mean_pred, earning_pred])

    return model;

#stocks = ['AMC', 'ANZ', 'BHP', 'CBA', 'NAB', 'GPT', 'CIM', 'CCL'];
stocks = ['CIM'];

priceIncrease=1.02
useDailyPriceChange = True
showPlot = False

for stock in stocks:
    print stock

    tradingData = readDataSets('/home/grant/Downloads/phd_data/' + stock + '2000-2012.csv', priceIncrease, useDailyPriceChange)
    
    epochSize = 60
    batchSize = 50
    batch_xs, batch_ys = tradingData.nextBatch(batchSize*epochSize)
    
    print batch_xs[0:50];
    
    batch_xs = numpy.reshape(batch_xs, [epochSize*batchSize, 100,1])
    batch_ys = numpy.reshape(batch_ys, [epochSize*batchSize,1, 1])
    
    #print batch_ys;
    
    model = buildModel();    
                        
    history = model.fit(batch_xs, batch_ys, epochs=50, batch_size=10, verbose = False)
    
    
    if showPlot:
        print(history.history.keys())
        plt.plot(history.history['acc'])
        plt.plot(history.history['loss'])
        plt.title('training accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['accuracy', 'loss'], loc='upper left')
        plt.yticks(np.arange(0.45, 0.55, step=0.05))
        plt.show()
    
    testData =  readDataSets('/home/grant/Downloads/phd_data/'+ stock +'2013-2017.csv', priceIncrease, useDailyPriceChange);
    
    
    testDataCount = 500;
    test_xs, test_ys = testData.nextBatch(testDataCount)
    test_xs = numpy.reshape(test_xs, [testDataCount,100, 1])
    test_ys = numpy.reshape(test_ys, [testDataCount,1, 1])
    
    score = model.evaluate(test_xs, test_ys, batch_size=50)
    
    print score
    
    pred_ys = model.predict(test_xs)
    
    print numpy.reshape( numpy.round(pred_ys)[0:50], [50])