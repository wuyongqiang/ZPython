# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
#from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import csv;
from util.ff1data import readDataSets;
import argparse
import sys
import numpy;

import tensorflow as tf;


FLAGS = None


def main(_):	
	
	#AMC, ANZ, BHP, CBA, NAB, GPT, CIM, CCL
	stock = "CCL"
	
	# Import data
	tradingData = readDataSets('/home/grant/Downloads/phd_data/'+stock+'2000-2012.csv')
	
	# Create the model
	x = tf.placeholder(tf.float32, [None, 100], "tradingPrices")
	W = tf.Variable(tf.zeros([100, 1]), name='priceWeight')
	b = tf.Variable(tf.zeros([1]), name='bias')
	y = tf.sigmoid(tf.matmul(x, W) + b)

	# Define loss and optimizer
	y_ = tf.placeholder(tf.float32, [None,1], "probHigherPrice")

	# The raw formulation of cross-entropy,
	#
	#	 tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
	#																 reduction_indices=[1]))
	#
	# can be numerically unstable.
	#
	# So here we use tf.losses.sparse_softmax_cross_entropy on the raw
	# outputs of 'y', and then average across the batch.
	#cross_entropy = tf.losses.mean_squared_error(y_, y)
	#cross_entropy = tf.losses.softmax_cross_entropy(y_,y);
	cross_entropy = tf.reduce_sum(tf.abs(tf.subtract(y_, y)))
	train_step = tf.train.GradientDescentOptimizer(0.9).minimize(cross_entropy)
	
	correct_prediction = tf.equal( tf.cast(tf.round(y), tf.int32), tf.cast(tf.round(y_), tf.int32))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	tf.summary.scalar('accuracy', accuracy)
	tf.summary.scalar('cross_entropy', cross_entropy)
	merged = tf.summary.merge_all()
	
	with tf.Session() as sess:
		train_writer = tf.summary.FileWriter('/tmp/train', sess.graph)
		tf.global_variables_initializer().run()
		# Train
		for i in range(60):
			batchSize = 50;
			batch_xs, batch_ys = tradingData.nextBatch(batchSize)
			batch_xs = numpy.reshape(batch_xs, [batchSize, 100])
			batch_ys = numpy.reshape(batch_ys, [batchSize, 1])
			sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})	
			summary, entropy = sess.run([merged,cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})		
			#print(entropy)
			#print (sess.run( cross_entropy , feed_dict={x: batch_xs, y_: batch_ys}))
			train_writer.add_summary(summary, i)	
		# Test trained model
		
		train_writer.close()
		
		testData = readDataSets('/home/grant/Downloads/phd_data/'+stock+'2013-2017.csv');
		test_xs, test_ys = testData.nextBatch(200)
		test_xs = numpy.reshape(test_xs, [200, 100])
		test_ys = numpy.reshape(test_ys, [200, 1])
		#print (test_xs)
		print(sess.run(tf.round(test_ys)))
		#print(sess.run(tf.abs(y), feed_dict={x: test_xs,y_:test_ys}))
		#print(sess.run(correct_prediction, feed_dict={x: test_xs,y_:test_ys}))
		print("acurracy")
		print(sess.run(accuracy, feed_dict={x: test_xs,y_:test_ys}))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='',
	help='Directory for storing input data')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run()