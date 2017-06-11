import tensorflow as tf

image_bytes = result.height * result.width * result.depth
record_bytes = label_bytes + image_bytes
#record_bytes为3073
reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
#每次读取的大小为3073
result.key, value = reader.read(filename)

# Convert from a string to a vector of uint8 that is record_bytes long.
record_bytes = tf.decode_raw(value, tf.uint8)

# The first bytes represent the label, which we convert from uint8->int32.
label = tf.cast(
    tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

# The remaining bytes after the label represent the image, which we reshape
# from [depth * height * width] to [depth, height, width].
depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                         [result.depth, result.height, result.width])
# Convert from [depth, height, width] to [height, width, depth].
uint8image = tf.transpose(depth_major, [1, 2, 0])