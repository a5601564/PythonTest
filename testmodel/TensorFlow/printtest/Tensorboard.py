import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# creat data
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

tf.tensorboard


output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))


def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size])) + 0.1
        Wx_plux_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plux_b
        else:
            outputs = activation_function(Wx_plux_b)
        return outputs


# creat data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1], 'x_input')
ys = tf.placeholder(tf.float32, [None, 1], 'y_input')

# create  in layers
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

# hidden layers


# out layers
prediction = add_layer(l1, 10, 1, activation_function=None)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
writer=tf.summary.FileWriter("logs/",graph=sess.graph)
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

for i in range(20000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass

        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)

        plt.pause(0.1)
