import tensorflow as tf
import numpy as np

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.randint(3, size=(100, 1))-1) # 随机输入
y_data = np.float32(np.random.randint(3, size=(100, 1))-1)

NUM_INPUTS = 1
NUM_HIDDEN = 5
NUM_OUTPUTS = 1
do_training=1

x = tf.placeholder(tf.float32, shape=[1], name='x')
y_ = tf.placeholder(tf.float32, shape=[1], name='y_')
#处理输入的结果 ：  -1  to 1
with tf.name_scope('layer1'):
    # initialize with a little noise and since we're using ReLU, we give them
    # a slightly positive bias
    #返回一个tensor其中的元素服从截断正态分布
    W_fc1=tf.truncated_normal([NUM_INPUTS],mean=0.5,stddev=0.707)
    W_fc1=tf.Variable(W_fc1,name='W_fc1')

    b_fc1=tf.truncated_normal([NUM_HIDDEN],mean=0.5,stddev=0.707)
    b_fc1=tf.Variable(b_fc1,name='b_fc1') #zaohua

    h_fc1 = tf.nn.relu(W_fc1 + b_fc1)

    #柱状图;
    tf.summary.histogram('W_fc1_summary',W_fc1)

#处理输出的结果 ：  -1  to 1
with tf.name_scope('layer2'):
    #矩阵函数 ---》 0 ， 1
    W_fc2 = tf.truncated_normal([NUM_OUTPUTS], mean=0.5, stddev=0.707)
    W_fc2 = tf.Variable(W_fc2, name='W_fc2')

    b_fc2 = tf.truncated_normal([NUM_OUTPUTS], mean=0.5, stddev=0.707)
    b_fc2 = tf.Variable(b_fc2, name='b_fc2')

    #y是个矩阵
    y = W_fc2+ b_fc2

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





sess = tf.InteractiveSession()


if do_training == 1:
    sess.run(tf.global_variables_initializer())

    for i in range(10001):
        if i%100 == 0:
            feed_dict={x: x_data[0], y_:y_data[0]}
            train_error = cross_entropy.eval(feed_dict)
            print("step %d, training error ==%.5f"%(i, train_error))
            if train_error < 0.0005:
                break


        sess.run(train_step, feed_dict=feed_dict)

    # test it out using the separate test data, though in this case it's
    # a bit silly since the test data is identical to the training data,
    # just in a different order.
    print("Test error using test data %g"
          %(cross_entropy.eval(feed_dict=feed_dict)))


print('\nCounting starting with: 0')
res = sess.run(results, feed_dict={x: [0]})
print('%.5f'%(res[0]))


print('---------------------------------------------------')
for i in range(10):
    res = sess.run(results, feed_dict={x: res})
    print('%.5f'%(res[0]))