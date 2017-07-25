import tensorflow as tf

import tools


class CNN(object):

    def __init__(self, learning_rate=1e-4):
        self.graph = tf.Graph()

        with self.graph.as_default():

            self.batch_input = tf.placeholder(tf.float32, shape=[None, 64, 64, 2], name='inputs')
            self.batch_target = tf.placeholder(tf.float32, shape=[None, 2], name='targets')

            conv1 = tf.layers.conv2d(
                inputs=self.batch_input,
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

            flatten = tools.layers.flatten(conv4)

            output = tf.layers.dense(
                inputs=flatten,
                units=2
            )

            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(tf.square(output - self.batch_target))

                self.summaries = {
                    'train': tf.summary.scalar('training', self.loss),
                    'val': tf.summary.scalar('validation', self.loss)
                }

            with tf.name_scope('train'):
                self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

            self.init = tf.global_variables_initializer()

    def train(self, train_generator, validation_data, max_steps, verbose=True, log_path=".logs/latest/"):

        with tf.Session(graph=self.graph) as sess:
            writer = tf.summary.FileWriter(log_path, graph=sess.graph)

            sess.run(self.init)

            for step in range(1, max_steps + 1):

                inputs, targets = train_generator.generate_batch(batch_size=100)

                sess.run(self.train_op, feed_dict={
                    self.batch_input: inputs,
                    self.batch_target: targets
                })

                if step % 25 == 0:

                    train_summary, train_loss = sess.run([self.summaries['train'], self.loss], feed_dict={
                        self.batch_input: inputs,
                        self.batch_target: targets
                    })

                    val_summary, val_loss = sess.run([self.summaries['val'], self.loss], feed_dict={
                        self.batch_input: validation_data[0],
                        self.batch_target: validation_data[1]
                    })

                    if verbose:
                        print('step: {} / loss: {:.4f} / val: {:.4f}'.format(step, train_loss, val_loss))

                    writer.add_summary(train_summary, step)
                    writer.add_summary(val_summary, step)
                    writer.flush()
