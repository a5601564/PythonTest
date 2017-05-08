import  os

do_training = 1 # 1 = do the training, 0 = load from file and just run it
save_trained = 1 # 1 = save to file after training, 0 = don't save
# Change the following to where you want the network to be saved to.
# Make sure to create the directory structure.
save_file = r'D:\javaweb\pythonDemo\testmodel\bincounter_TensorFlow_example-master\tflogdir/bincounter.ckpt'
write_for_tensorboard = 0 # 1 = write info for TensorBoard, 0 = don't
# Change the following to where you want the info to be saved.
# Make sure to create the directory structure.
tensorboard_file = r'D:\javaweb\pythonDemo\testmodel\bincounter_TensorFlow_example-master\tflogdir/bincounter_tb/1'


NUM_INPUTS = 3
NUM_HIDDEN = 5
NUM_OUTPUTS = 3
NUM_IN_TRAINING_SET = 8
NUM_IN_TEST_SET = 8

inputvals  = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1],
              [1, 1, 0], [1, 1, 1]]
targetvals = [[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0],
              [1, 1, 1], [0, 0, 0]]
testinputs  = [[0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 1],
               [1, 0, 0], [0, 0, 0], [1, 1, 0]]
testtargets = [[0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 1, 0],
               [1, 0, 1], [0, 0, 1], [1, 1, 1]]

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, NUM_INPUTS], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, NUM_OUTPUTS], name='y_')

#处理输入的结果 ： 0 0 0
with tf.name_scope('layer1'):
    # initialize with a little noise and since we're using ReLU, we give them
    # a slightly positive bias
    #返回一个tensor其中的元素服从截断正态分布
    W_fc1=tf.truncated_normal([NUM_INPUTS,NUM_HIDDEN],mean=0.5,stddev=0.707)
    W_fc1=tf.Variable(W_fc1,name='W_fc1')

    b_fc1=tf.truncated_normal([NUM_HIDDEN],mean=0.5,stddev=0.707)
    b_fc1=tf.Variable(b_fc1,name='b_fc1')

    h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

    tf.summary.histogram('W_fc1_summary',W_fc1)

#处理输出的结果 ： 0 0 1
with tf.name_scope('layer2'):
    #矩阵函数 ---》 0 ， 1
    W_fc2 = tf.truncated_normal([NUM_HIDDEN, NUM_OUTPUTS], mean=0.5, stddev=0.707)
    W_fc2 = tf.Variable(W_fc2, name='W_fc2')

    b_fc2 = tf.truncated_normal([NUM_OUTPUTS], mean=0.5, stddev=0.707)
    b_fc2 = tf.Variable(b_fc2, name='b_fc2')

    #y是个矩阵
    y = tf.matmul(h_fc1, W_fc2) + b_fc2

    results = tf.sigmoid(y, name='results')# 使结果变为0-1 之间

    tf.summary.histogram('W_fc2_summary', W_fc2)
    tf.summary.histogram('b_fc2_summary', b_fc2)
    tf.summary.histogram('y_summary', y)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_))
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.RMSPropOptimizer(0.25, momentum=0.5).minimize(cross_entropy)


if write_for_tensorboard == 1:
    merged_summary = tf.summary.merge_all()
    print("Writing for TensorBoard to file %s"%(tensorboard_file))
    writer = tf.summary.FileWriter(tensorboard_file)
    writer.add_graph(sess.graph)

if do_training == 1:
    sess.run(tf.global_variables_initializer())

    for i in range(10001):
        if i%100 == 0:
            feed_dict={x: inputvals, y_:targetvals}
            train_error = cross_entropy.eval(feed_dict)
            print("step %d, training error ==%.5f"%(i, train_error))
            if train_error < 0.0005:
                break

        if write_for_tensorboard == 1 and i%5 == 0:
            s = sess.run(merged_summary, feed_dict={x: inputvals, y_:targetvals})
            writer.add_summary(s, i)

        sess.run(train_step, feed_dict={x: inputvals, y_: targetvals})

    # test it out using the separate test data, though in this case it's
    # a bit silly since the test data is identical to the training data,
    # just in a different order.
    print("Test error using test data %g"
          %(cross_entropy.eval(feed_dict={x: testinputs, y_: testtargets})))

    if save_trained == 1:
        print("Saving neural network to %s.*"%(save_file))
        saver = tf.train.Saver()
        saver.save(sess, save_file)

else: # if we're not training then we must be loading from file

    print("Loading neural network from %s"%(save_file))
    saver = tf.train.Saver()
    saver.restore(sess, save_file)
    # Note: the restore both loads and initializes the variables

print('\nCounting starting with: 0 0 0')
res = sess.run(results, feed_dict={x: [[0, 0, 0]]})
print('%.5f %.5f %.5f'%(res[0][0], res[0][1], res[0][2]))


print('---------------------------------------------------')
for i in range(10):
    res = sess.run(results, feed_dict={x: res})
    print('%.5f %.5f %.5f'%(res[0][0], res[0][1], res[0][2]))
