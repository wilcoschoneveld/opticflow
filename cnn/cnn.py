import time
import tensorflow as tf

import data.generator


input_batch = tf.placeholder(tf.float32, shape=[None, 64, 64, 2], name='inputs')
target_batch = tf.placeholder(tf.float32, shape=[None, 2], name='targets')

conv1 = tf.layers.conv2d(
    inputs=input_batch,
    filters=32,
    kernel_size=3,
    strides=2,
    activation=tf.nn.relu,
    name='conv1'
)

conv2 = tf.layers.conv2d(
    inputs=conv1,
    filters=64,
    kernel_size=3,
    strides=2,
    activation=tf.nn.relu,
    name='conv2'
)

conv3 = tf.layers.conv2d(
    inputs=conv2,
    filters=128,
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
    loss = tf.reduce_mean(tf.square(output - target_batch))
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("tmp/log", graph=tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    i = 0

    start = time.time()

    gen = data.generator.DataGenerator('data/images/train/*', image_size=64, max_flow=5, max_scale=5, noise_level=5)

    for epoch in range(1, 500):

        for j in range(50):
            inputs, targets = gen.generate_batch(batch_size=100)

            _, loss_value = sess.run([train_op, loss], feed_dict={input_batch: inputs, target_batch: targets})

        print('epoch: {} / loss: {:.4f} / time: {}'.format(epoch, loss_value, time.time() - start))

        summary = sess.run(merged, feed_dict={input_batch: inputs, target_batch: targets})
        writer.add_summary(summary, epoch)
