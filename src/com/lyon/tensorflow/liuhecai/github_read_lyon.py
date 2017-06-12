import tensorflow as tf
import numpy as np


fileName = 'source/2000.csv'

try_epochs = 1
batch_size = 8

F = 6 # this is the list of your features
L = 1 # this is one-hot vector of 3 representing the label

# set defaults to something (TF requires defaults for the number of cells you are going to read)
rDefaults = [['a'] for row in range((F+L))]

# function that reads the input file, line-by-line
def read_from_csv(filename_queue):
    reader = tf.TextLineReader() # skipt the header line
    _, csv_row = reader.read(filename_queue) # read one line
    data = tf.decode_csv(csv_row, record_defaults=rDefaults) # use defaults for this line (in case of missing data)
    features = tf.string_to_number(tf.slice(data, [0], [F]), tf.float32) # cells 1-2 is the list of features
    label = tf.string_to_number(tf.slice(data, [F], [L]), tf.float32) # the remainin 3 cells is the list for one-hot label
    return features, label

# function that packs each read line into batches of specified size
def input_pipeline(fName, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        [fName],
        num_epochs=num_epochs,
        shuffle=True)  # this refers to multiple files, not line items within files
    features, label = read_from_csv(filename_queue)
    min_after_dequeue = 10000 # min of where to start loading into memory
    capacity = min_after_dequeue + 3 * batch_size # max of how much to load into memory
    # this packs the above lines into a batch of size you specify:
    feature_batch, label_batch = tf.train.shuffle_batch(
        [features, label],
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return feature_batch, label_batch


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
# these are the student label, features, and label:
features, labels = input_pipeline(fileName, batch_size, try_epochs)



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


init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    gInit = tf.global_variables_initializer().run()
    lInit = tf.local_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop():
            # load student-label, features, and label as a batch:
            feature_batch, label_batch = sess.run([features, labels])

            print(feature_batch);
            print(label_batch);
            print('----------');

    except tf.errors.OutOfRangeError:
        print("Finish looping through the file")

    finally:
        coord.request_stop()

    coord.join(threads)

    for step in range(3000):  #
        feed_dict={train_X:feature_batch, train_Y:label_batch}
        sess.run(train_op,feed_dict=feed_dict)

        if step %100==0:
            _, y_predict_value = sess.run([train_op,prediction],feed_dict=feed_dict)
            print(y_predict_value)

