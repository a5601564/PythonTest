# coding=utf-8
import tensorflow as tf
import numpy as np

def read_my_file_format(filename_queue):
    """从文件名队列读取一行数据

    输入：
    -----
    filename_queue：文件名队列，举个例子，可以使用`tf.train.string_input_producer(["file0.csv", "file1.csv"])`方法创建一个包含两个CSV文件的队列

    输出：
    -----
    一个样本：`[features, label]`
    """
    reader = tf.SomeReader()  # 创建Reader
    key, record_string = reader.read(filename_queue)  # 读取一行记录
    example, label = tf.some_decoder(record_string)  # 解析该行记录
    processed_example = some_processing(example)  # 对特征进行预处理
    return processed_example, label

def input_pipeline(filenames, batch_size, num_epochs=None):
    """ 从一组文件中读取一个批次数据

    输入：
    -----
    filenames：文件名列表，如`["file0.csv", "file1.csv"]`
    batch_size：每次读取的样本数
    num_epochs：每个文件的读取次数

    输出：
    -----
    一批样本，`[[example1, label1], [example2, label2], ...]`
    """
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)  # 创建文件名队列
    example, label = read_my_file_format(filename_queue)  # 读取一个样本
    # 将样本放进样本队列，每次输出一个批次样本
    #   - min_after_dequeue：定义输出样本后的队列最小样本数，越大随机性越强，但start up时间和内存占用越多
    #   - capacity：队列大小，必须比min_after_dequeue大
    min_after_dequeue = 10000
    capacity = min_after_dqueue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch

def main(_):
    x, y = input_pipeline(['file0.csv', 'file1.csv'], 1000, 5)
    train_op = some_func(x, y)
    init_op = tf.global_variables_initializer()
    local_init_op = tf.local_variables_initializer()  # local variables like epoch_num, batch_size
    sess = tf.Session()

    sess.run(init_op)
    sess.run(local_init_op)

    # `QueueRunner`用于创建一系列线程，反复地执行`enqueue` op
    # `Coordinator`用于让这些线程一起结束
    # 典型应用场景：
    #   - 多线程准备样本数据，执行enqueue将样本放进一个队列
    #   - 一个训练线程从队列执行dequeu获取一批样本，执行training op
    # `tf.train`的许多函数会在graph中添加`QueueRunner`对象，如`tf.train.string_input_producer`
    # 在执行training op之前，需要保证Queue里有数据，因此需要先执行`start_queue_runners`
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            sess.run(train_op)
    except tf.errors.OutOfRangeError:
        print 'Done training -- epoch limit reached'
    finally:
        coord.request_stop()

    # Wait for threads to finish
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    tf.app.run()