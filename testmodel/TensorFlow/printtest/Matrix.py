import tensorflow as tf
import numpy as np

#creat data
matrix1=tf.constant([[3,3]])
matrix2=tf.constant([[2],[2]])

product=tf.matmul(matrix1,matrix2)

with tf.Session() as sess:
    result=sess.run(product)
    print(result)