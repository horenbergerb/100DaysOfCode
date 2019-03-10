from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0)

total = a+b

print(a)
print(b)
print(total)

#writer = tf.summary.FileWriter('.')
#writer.add_graph(tf.get_default_graph())
#writer.flush()

sess = tf.Session()
print(sess.run(total))
print(sess.run({'ab':(a,b), 'total':total}))

#For each call to Session.run, all tensors will only have a single value!!

#feeding inputs:
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

z = x+y

print(sess.run(z, feed_dict={x: [1,3], y:[2,4]}))

#datasets!

my_data = [
    [1,2,],
    [2,3,],
    [6,7,],
]

slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()

#r = tf.random_normal([10,3])
#dataset = tf.data.Dataset.from_tensor_slices(r)
#iterator = dataset.make_initializable_iterator()
#next_row = iterator.get_next()

while True:
    try:
        print(sess.run(next_item))
    except tf.errors.OutOfRangeError:
        break
