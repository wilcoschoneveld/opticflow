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

    loss_train = tf.summary.scalar('training', loss)
    loss_val = tf.summary.scalar('validation', loss)

with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

writer = tf.summary.FileWriter("tmp/log", graph=tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    i = 0

    start = time.time()

    val_inputs, val_targets = data.generator.DataGenerator(
        pattern='data/images/test/city.jpg',
        image_size=64,
        max_flow=5,
        max_scale=5,
        noise_level=5,
        interp='bicubic').generate_batch(batch_size=1000)

    gen = data.generator.DataGenerator(
        pattern='data/images/train/*',
        image_size=64,
        max_flow=5,
        max_scale=5,
        noise_level=5,
        interp='bicubic')

    for step in range(1, 25000):

        inputs, targets = gen.generate_batch(batch_size=100)

        sess.run(train_op, feed_dict={input_batch: inputs, target_batch: targets})

        if step % 25 == 0:

            train_loss, train_loss_value = sess.run([loss_train, loss], feed_dict={input_batch: inputs, target_batch: targets})

            val_loss, val_loss_value = sess.run([loss_val, loss], feed_dict={input_batch: val_inputs, target_batch: val_targets})

            print('step: {} / loss: {:.4f} / val: {:.4f} / time: {}'.format(step, train_loss_value, val_loss_value,
                                                                            time.time() - start))

            writer.add_summary(train_loss, step)
            writer.add_summary(val_loss, step)
            writer.flush()
