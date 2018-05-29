from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Activation
from util.ff1data import readDataSets
import numpy
from keras.optimizers import RMSprop, SGD;
import tensorflow as tf;


import matplotlib;

#matplotlib.use('Agg')

import matplotlib.pyplot as plt;


model = Sequential()

#lstm
#model.add(LSTM( 1, return_sequences=True, activation='linear', input_shape=(20,5)))

#model.add(LSTM( 3, return_sequences=True, activation='linear'))

#model.add(LSTM( 1, return_sequences=True, activation='linear'))

#model.add(Dense(5, input_shape=(20,5)))

#model.add(Dense(1 ))
#model.add(Dropout(0.5))

#multi fully connected

model.add(Dense(200, kernel_initializer='lecun_uniform', activation='relu', input_shape=(None,100)))

#model.add(Dense(1000,kernel_initializer='lecun_uniform',activation='relu'))

#model.add(Dense(200,kernel_initializer='lecun_uniform',activation='relu'))

model.add(Dense(50,kernel_initializer='lecun_uniform',activation='relu'))

model.add(Dense(20,kernel_initializer='lecun_uniform',activation='relu'))

model.add(Dense(10,kernel_initializer='lecun_uniform',activation='relu'))

model.add(Dense(1))

#model.add(Activation('sigmoid'))

def mean_pred(y_true, y_pred):
    correct_prediction = tf.equal( tf.cast(tf.round(y_true), tf.int32), tf.cast(tf.round(y_pred), tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def earning_pred(y_true, y_pred):
    positive_pred = tf.reduce_sum(tf.cast(tf.round(y_pred), tf.int32));
    correct_prediction = tf.equal( tf.cast(tf.round(y_true), tf.int32), tf.cast(tf.round(y_pred), tf.int32))
    positive_correct = tf.logical_and( correct_prediction, tf.greater(y_true, tf.constant(0.5, dtype=tf.float32)))
    return tf.cond( tf.greater(positive_pred, tf.constant(0)), lambda:tf.reduce_sum(tf.to_int32( positive_correct)) / positive_pred, lambda:tf.to_int32(0) );
   
     
     
myOptimizer = SGD(lr=0.01, clipvalue=0.5);
model.compile(loss='mean_absolute_error', optimizer=myOptimizer, metrics=['accuracy', mean_pred, earning_pred])


#AMC(49.40%), ANZ(65%), BHP(63), CBA(67%), NAB(59%0, GPT(57.6%), CIM(57.8%), CCL(62.2%)
#0.473,0.572,0.493,0.616,0.555,0.535,0.609,0.414
stock = "ANZ"

tradingData = readDataSets('/home/grant/Downloads/phd_data/' + stock + '2000-2012.csv')

dataCount = 3000
dimesion = 1
batch_xs, batch_ys = tradingData.nextBatch(dimesion*dataCount)

batch_xs = numpy.reshape(batch_xs, [dataCount,dimesion, 100])
batch_ys = numpy.reshape(batch_ys, [dataCount,dimesion, 1])

#print batch_ys;
                    
history = model.fit(batch_xs, batch_ys, epochs=20, batch_size=50, verbose=0)
print(history.history.keys())

#plt.plot(history.history['acc'])
#plt.plot(history.history['loss'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'loss'], loc='upper left')
#plt.show()

testData =  readDataSets('/home/grant/Downloads/phd_data/'+ stock +'2013-2017.csv');

overall_acc = 0.0;
for x in range(0, 25):
    testDataCount = 20;
    test_xs, test_ys = testData.nextBatch(testDataCount)
    test_xs = numpy.reshape(test_xs, [testDataCount,1, 100])
    test_ys = numpy.reshape(test_ys, [testDataCount,1, 1])

    score = model.evaluate(test_xs, test_ys, batch_size=5, verbose=0)

    print score[0];
    overall_acc += score[0]

    pred_ys = model.predict(test_xs)

    #print pred_ys
    
    model.fit(test_xs, test_ys, epochs=1, batch_size=5, verbose=0)

print pred_ys

print "-------------------------------------------------------------------"    
print overall_acc/25.0;
print "end"

    