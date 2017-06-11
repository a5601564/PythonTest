import tensorflow as tf
import numpy as np


fileName = 'source/Student_grades.csv'

try_epochs = 1
batch_size = 8

F = 2 # this is the list of your features
L = 3 # this is one-hot vector of 3 representing the label

# set defaults to something (TF requires defaults for the number of cells you are going to read)
rDefaults = [['a'] for row in range((F+L))]

# function that reads the input file, line-by-line
def read_from_csv(filename_queue):
    reader = tf.TextLineReader(skip_header_lines=True) # skipt the header line
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

# these are the student label, features, and label:
features, labels = input_pipeline(fileName, batch_size, try_epochs)

def grade (index):
    if index==[0]:
        label='A'
    elif index==[1]:
        label='B'
    elif index==[2]:
        label='C'
    return label

x = tf.placeholder(tf.float32, [None, 2])

W = tf.Variable(tf.zeros([2, 3]))

b = tf.Variable(tf.zeros([3]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 3])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

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
        sess.run(train_step,feed_dict={x:feature_batch, y_:label_batch})

        if step %100==0:
            print(step, sess.run(cross_entropy, feed_dict={x:feature_batch, y_:label_batch}))
            print("W =", sess.run(W))
            print("b =", sess.run(b))

            student9=sess.run(y,feed_dict={x:[[11, 7]]})
            print("Student 9:", student9, grade(sess.run(tf.argmax(student9,1))))

            student10 = sess.run(y, feed_dict={x: [[3, 4]]})
            print("Student 10:", student10, grade(sess.run(tf.argmax(student10, 1))))

            student11 = sess.run(y, feed_dict={x: [[1, 0]]})
            print("Student 11:", student11, grade(sess.run(tf.argmax(student11, 1))))