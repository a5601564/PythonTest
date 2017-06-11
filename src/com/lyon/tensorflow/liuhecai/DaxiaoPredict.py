from __future__ import print_function,division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl


#read file
filename_queue = tf.train.string_input_producer(["source/2000.csv"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1], [1], [1], [1], [1], [1], [1]]

col1, col2, col3, col4, col5, col6, col7 = tf.decode_csv(value, record_defaults=record_defaults)
features = tf.stack([col1, col2, col3, col4, col5, col6, col7])



#method
def judge(train_Tema):
    if train_Tema>=25:
        return 1.0;
    else:
        return 0.0;

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    # 区别：大框架，定义层 layer，里面有 小部件
    with tf.name_scope('layer'):
        # 区别：小部件
        with tf.name_scope('weights'):
            Weights = tf.Variable([1.0], name="weight")
        with tf.name_scope('biases'):
            biases = tf.Variable([1.0], name="bias")
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.multiply(train_X, Weights)+biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        return outputs

# Prepare train data
train_X = tf.placeholder("float32")
train_Y = tf.placeholder("float32")
# pl.plot(train_X,train_Y)
# pl.show()
# Define the model


y_predict = add_layer(train_X, 7, 49, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(y_predict, 49, 1, activation_function=None)

tf.summary.histogram('prediction', prediction)

with tf.name_scope('loss'):
    loss= tf.square(train_Y -prediction)

# 区别：定义框架 train
with tf.name_scope('train'):
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)




with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    list_y=[]

    for i in range(1200):
        # Retrieve a single instance:
        example, tema = sess.run([features, col7])
        #result=judge(tema)
        feed_dict={train_X: example,train_Y: tema}
        _, y_predict_value = sess.run([train_op,prediction],feed_dict=feed_dict)
        list_y.append(y_predict_value)
        print(list_y)

    coord.request_stop()
    coord.join(threads)