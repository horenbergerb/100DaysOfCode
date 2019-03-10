import tensorflow as tf
import numpy as np

x = tf.constant([[1],[2],[3],[4]], dtype=tf.float32)
y_true = tf.constant([[0],[-1],[-2],[-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)

y_pred = linear_model(x)

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

print(sess.run(y_pred))

loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

for i in range(1000):
    #first part seems to absorb some extra data about types
    _, loss_value = sess.run((train, loss))

print(sess.run(y_pred))
