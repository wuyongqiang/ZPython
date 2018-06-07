from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Activation
from util.ff1data import readDataSets
import numpy
from keras.optimizers import RMSprop
import tensorflow as tf;

model = Sequential()

model.add(LSTM( 10, return_sequences=True, activation='linear', input_shape=(20,5)))

model.add(LSTM( 10, return_sequences=True, activation='linear'))

model.add(LSTM( 1, return_sequences=True, activation='linear'))

#model.add(Dense(5, input_shape=(20,5)))

model.add(Dense(1 ))
#model.add(Dropout(0.5))
model.add(Activation('tanh'))

def mean_pred(y_true, y_pred):
    correct_prediction = tf.equal( tf.cast(tf.round(y_true), tf.int32), tf.cast(tf.round(y_pred), tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def earning_pred(y_true, y_pred):
    positive_pred = tf.reduce_sum(tf.cast(tf.round(y_pred), tf.int32));
    correct_prediction = tf.equal( tf.cast(tf.round(y_true), tf.int32), tf.cast(tf.round(y_pred), tf.int32))
    positive_correct = tf.logical_and( correct_prediction, tf.greater(y_true, tf.constant(0.5, dtype=tf.float32)))
    return tf.cond( tf.greater(positive_pred, tf.constant(0)), lambda:tf.reduce_sum(tf.to_int32( positive_correct)) / positive_pred, lambda:tf.to_int32(0) );
   
     
     
rmsProp= RMSprop(lr=0.01);
model.compile(loss='mean_absolute_error', optimizer=rmsProp, metrics=[mean_pred, earning_pred])

#AMC, ANZ, BHP, CBA, NAB, GPT, CIM, CCL
stock = "CCL"

tradingData = readDataSets('/home/grant/Downloads/phd_data/' + stock + '2000-2012.csv')

epochSize = 60
batchSize = 50
batch_xs, batch_ys = tradingData.nextLstmBatch(batchSize*epochSize)

batch_xs = numpy.reshape(batch_xs, [batchSize*epochSize/20, 20, 5])
batch_ys = numpy.reshape(batch_ys, [batchSize*epochSize/20, 20, 1])

#print batch_ys;
                    
model.fit(batch_xs, batch_ys, epochs=10, batch_size=batchSize)

testData =  readDataSets('/home/grant/Downloads/phd_data/'+ stock +'2013-2017.csv');


testDataCount = 500;
test_xs, test_ys = testData.nextLstmBatch(testDataCount)
test_xs = numpy.reshape(test_xs, [testDataCount/20, 20, 5])
test_ys = numpy.reshape(test_ys, [testDataCount/20, 20, 1])

score = model.evaluate(test_xs, test_ys, batch_size=testDataCount)

print score

pred_ys = model.predict(test_xs)

print pred_ys