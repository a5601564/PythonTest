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
X = tf.placeholder("float")
Y = tf.placeholder("float")
w = tf.Variable(1.0, name="weight")
b = tf.Variable(1.0, name="bias")



y_predict= tf.nn.relu(tf.multiply(X, w)+b)
loss = tf.square(Y -y_predict)

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)



# Create session to run
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    list_y=[]
    epoch = 1
    for i in range(10):

        for (x, y) in zip(train_X, train_Y):
            feed_dict={X: x,Y: y}
            _, y_predict_value = sess.run([train_op,y_predict],feed_dict=feed_dict)
            if i==9:
                list_y.append(y_predict_value)
                #print(feed_dict)
        print("Epoch: {}, y_predict_value: {},".format(epoch, y_predict_value))

        epoch += 1


#draw
plt.plot(train_X,train_Y,"+")
plt.plot(train_X,list_y)
plt.show()

