'''
Created on Jan 28, 2018

@author: grant
'''

import tensorflow as tf
from tensorflow.python.ops import math_ops, array_ops
from tensorflow.python.framework import dtypes


if __name__ == '__main__':
    a = tf.constant(1.2)
    b = tf.constant(2.4)
    sess = tf.Session()
    print(sess.run([a,b]));
    
    devices = sess.list_devices()
    for d in devices:
        print(d.name)
  
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(r1, c)
    
    h = sess.partial_run_setup([r1, r2], [a,b,c])
    res = sess.partial_run(h, r1, feed_dict={a:3,b:4})
    print(res)
    
    h = sess.partial_run_setup([r2, r1], [a,b,c])
    res = sess.partial_run(h, r2, feed_dict={a:3,b:5, c:2})
  
    print(res)
    
    print(sess.run([r1, r2, a, b, c], feed_dict={a: 1, b: 2, c:3}))
    
    norm = tf.random_normal([3, 3], seed=1234)

    y=[1, 2, 3, 8, 7, 9];
    
    print("reduce_mean")
    mean = sess.run(tf.reduce_mean(y)); 
    print(mean);
    
    print(sess.run(tf.square(y - mean)))
    
    print(sess.run(tf.argmax(y)));
    
    print(sess.run(norm));
    
    train_writer = tf.summary.FileWriter('/tmp/train', sess.graph)
    train_writer.add_summary(sess.run(tf.summary.scalar("test", tf.reduce_mean(tf.square(y - mean)))))
    
    
    
    sess.close()