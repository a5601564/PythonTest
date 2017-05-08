# -*- coding: utf-8 -*- 

from __future__ import print_function,division
import tensorflow as tf

#building the graph

#Create a Variable, that will be initialized to the scalar value 0.
state=tf.Variable(0,name="state")
print("the name of this variable:",state.name)

# Create an Op to add 1 to `state`.
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# Variables must be initialized by running an `init` Op after having
# launched the graph.  We first have to add the `init` Op to the graph.
init_op = tf.initialize_all_variables()

# Launch the graph and run the ops.
with tf.Session() as sess:
    # Run the 'init' op
    sess.run(init_op)
    # Print the initial value of 'state'
    print(sess.run(state))
    # Run the op that updates 'state' and print 'state'.
    for _ in range(3):
        sess.run(update)
        print("value of state:",sess.run(state))