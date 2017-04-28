import tensorflow as tf

filenames = ['a.csv']
filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
example, label = tf.decode_csv(value, record_defaults=[['null'], ['null']])
# 使用tf.train.batch()会多加了一个样本队列和一个QueueRunner。Decoder解后数据会进入这个队列，再批量出队。
# 虽然这里只有一个Reader，但可以设置多线程，相应增加线程数会提高读取速度，但并不是线程越多越好。
example_batch, label_batch = tf.train.batch(
    [example, label], batch_size=5)
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(10):
        print (example_batch.eval())
    coord.request_stop()
    coord.join(threads)