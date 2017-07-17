import time
import tools.layers

import tensorflow as tf
import data

input1 = tf.placeholder(tf.float32, shape=[None, 64, 64, 1], name='inputs1')
conv1 = tf.layers.conv2d(input1, 32, 7, 2, activation=tf.nn.relu, name='conv1')
conv2 = tf.layers.conv2d(conv1, 64, 5, 2, activation=tf.nn.relu, name='conv2')
conv3 = tf.layers.conv2d(conv2, 128, 3, 2, activation=tf.nn.relu, name='conv3')
conv4 = tf.layers.conv2d(conv3, 128, 3, 2, activation=tf.nn.relu, name='conv4')
flat = tools.layers.flatten(conv4)
hidden1 = tf.layers.dense(flat, 100, tf.nn.relu, name='hidden')

input2 = tf.placeholder(tf.float32, shape=[None, 64, 64, 1], name='inputs2')
conv1 = tf.layers.conv2d(input2, 32, 7, 2, activation=tf.nn.relu, name='conv1', reuse=True)
conv2 = tf.layers.conv2d(conv1, 64, 5, 2, activation=tf.nn.relu, name='conv2', reuse=True)
conv3 = tf.layers.conv2d(conv2, 128, 3, 2, activation=tf.nn.relu, name='conv3', reuse=True)
conv4 = tf.layers.conv2d(conv3, 128, 3, 2, activation=tf.nn.relu, name='conv4', reuse=True)
flat = tools.layers.flatten(conv4)
hidden2 = tf.layers.dense(flat, 100, tf.nn.relu, name='hidden', reuse=True)

stacked = tf.concat([hidden1, hidden2], 1)

hidden = tf.layers.dense(inputs=stacked, units=200, activation=tf.nn.relu, name='hidden2')

output = tf.layers.dense(inputs=hidden, units=2, name='output')


output_batch = tf.placeholder(tf.float32, shape=[None, 2], name='outputs')

with tf.name_scope('loss'):
    loss_op = tf.reduce_mean(tf.square(output - output_batch))

with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss_op)

writer = tf.summary.FileWriter("tmp/log", graph=tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    i = 0

    start = time.time()

    for _in, _out in data.generate_batches(10, 64):
        _in1 = _in[:, :, :, 0:1]
        _in2 = _in[:, :, :, 1:2]
        _, loss = sess.run([train_op, loss_op], feed_dict={input1: _in1, input2: _in2, output_batch: _out})

        print('i: {} / loss: {:.4f} / time: {}'.format(i, loss, time.time() - start))

        i += 1
