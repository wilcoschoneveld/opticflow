import time

import tensorflow as tf
import data_old

input_batch = tf.placeholder(tf.float32, shape=[None, 64, 64, 2], name='inputs')
output_batch = tf.placeholder(tf.float32, shape=[None, 2], name='outputs')

conv1 = tf.layers.conv2d(
    inputs=input_batch,
    filters=64,
    kernel_size=7,
    strides=2,
    activation=tf.nn.relu,
    name='conv1'
)

conv2 = tf.layers.conv2d(
    inputs=conv1,
    filters=128,
    kernel_size=5,
    strides=2,
    activation=tf.nn.relu,
    name='conv2'
)

conv3 = tf.layers.conv2d(
    inputs=conv2,
    filters=256,
    kernel_size=3,
    activation=tf.nn.relu,
    name='conv3'
)

conv4 = tf.layers.conv2d(
    inputs=conv3,
    filters=256,
    kernel_size=3,
    strides=2,
    activation=tf.nn.relu,
    name='conv4'
)

flatten = tf.contrib.layers.flatten(inputs=conv4, scope='flatten')

output = tf.layers.dense(
    inputs=flatten,
    units=2
)

with tf.name_scope('loss'):
    loss_op = tf.reduce_mean(tf.square(output - output_batch))

with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss_op)

writer = tf.summary.FileWriter("tmp/log", graph=tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    i = 0

    start = time.time()

    for _in, _out in data_old.generate_batches(10, 64):
        _, loss = sess.run([train_op, loss_op], feed_dict={input_batch: _in, output_batch: _out})

        print('i: {} / loss: {:.4f} / time: {}'.format(i, loss, time.time() - start))

        i += 1
