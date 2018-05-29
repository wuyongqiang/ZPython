from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Activation
from util.ff1data import readDataSets
import numpy
from keras.optimizers import RMSprop, SGD;
import tensorflow as tf;
import numpy as np;

import matplotlib;


import matplotlib.pyplot as plt;


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
    
    model.add(Dense(2000, kernel_initializer='lecun_uniform', input_shape=(None,100)))
    
    model.add(Dense(200,kernel_initializer='lecun_uniform', activation='relu'))
    
    model.add(Dense(100,kernel_initializer='lecun_uniform', activation='softmax'))
    
    model.add(Dense(50,kernel_initializer='lecun_uniform', activation='relu'))
    
    model.add(Dense(20,kernel_initializer='lecun_uniform', activation='softmax'))
    
    model.add(Dense(10,kernel_initializer='lecun_uniform', activation='relu'))
    
    model.add(Dense(1, kernel_initializer='lecun_uniform'))
    
    #model.add(Activation('softmax'))
    
    myOptimizer = SGD(lr=0.01, clipvalue=0.5);
    model.compile(loss='mean_absolute_error', optimizer=myOptimizer, metrics=['accuracy', mean_pred, earning_pred])

    return model;

stocks = ['AMC', 'ANZ', 'BHP', 'CBA', 'NAB', 'GPT', 'CIM', 'CCL'];

priceIncrease=1.02

for stock in stocks:
    print stock

    tradingData = readDataSets('/home/grant/Downloads/phd_data/' + stock + '2000-2012.csv', priceIncrease)
    
    epochSize = 60
    batchSize = 50
    batch_xs, batch_ys = tradingData.nextBatch(batchSize*epochSize)
    
    batch_xs = numpy.reshape(batch_xs, [epochSize,batchSize, 100])
    batch_ys = numpy.reshape(batch_ys, [epochSize,batchSize, 1])
    
    #print batch_ys;
    
    model = buildModel();    
                        
    history = model.fit(batch_xs, batch_ys, epochs=20, batch_size=50, verbose = False)
    
    showPlot = False
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
    
    testData =  readDataSets('/home/grant/Downloads/phd_data/'+ stock +'2013-2017.csv', priceIncrease);
    
    
    testDataCount = 500;
    test_xs, test_ys = testData.nextBatch(testDataCount)
    test_xs = numpy.reshape(test_xs, [testDataCount/50,50, 100])
    test_ys = numpy.reshape(test_ys, [testDataCount/50,50, 1])
    
    score = model.evaluate(test_xs, test_ys, batch_size=testDataCount)
    
    print score
    
    #pred_ys = model.predict(test_xs)
    
    #print pred_ys