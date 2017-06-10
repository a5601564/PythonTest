from __future__ import print_function,division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

# Prepare train data
train_X= x=np.linspace(0,4*np.pi,100)

train_Y = np.sin(train_X)
# pl.plot(train_X,train_Y)
# pl.show()
# Define the model

# define placeholder for inputs to network
# 区别：大框架，里面有 inputs x，y
with tf.name_scope('inputs'):
    X= tf.placeholder("float")
    Y = tf.placeholder("float")
    w = tf.Variable(1.0, name="weight")
    b = tf.Variable(1.0, name="bias")


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    # 区别：大框架，定义层 layer，里面有 小部件
    with tf.name_scope('layer'):
        # 区别：小部件
        with tf.name_scope('weights'):
            Weights = tf.Variable(1.0, name="weight")
        with tf.name_scope('biases'):
            biases = tf.Variable(1.0, name="bias")
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.multiply(X, Weights)+biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        return outputs

# y_predict= tf.nn.relu(tf.multiply(X, w)+b)
# loss = tf.square(Y -y_predict)
#
# train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# add hidden layer

y_predict = add_layer(X, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(y_predict, 10, 1, activation_function=None)

tf.summary.histogram('prediction', prediction)
# the error between prediciton and real data
# 区别：定义框架 loss
with tf.name_scope('loss'):
    loss= tf.square(Y -prediction)



# 区别：定义框架 train
with tf.name_scope('train'):
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)


total_step = 0
# Create session to run
with tf.Session() as sess:
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('/tmp/mnist_logs', sess.graph)
    sess.run(tf.initialize_all_variables())
    list_y=[]
    epoch = 1
    for i in range(10):
        total_step += 1
        for (x, y) in zip(train_X, train_Y):
            feed_dict={X: x,Y: y}
            _, y_predict_value = sess.run([train_op,prediction],feed_dict=feed_dict)
            if i==9:
                list_y.append(y_predict_value)
                #print(feed_dict)


            summary_str = sess.run(merged_summary_op)
            summary_writer.add_summary(summary_str, total_step)
        print("Epoch: {}, y_predict_value: {},".format(epoch, y_predict_value))

        epoch += 1


#draw
plt.plot(train_X,train_Y,"+")
plt.plot(train_X,list_y)
plt.show()

