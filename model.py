import tensorflow as tf

import tools


class CNN(object):

    @staticmethod
    def create_conv_layers(input, reuse=False):
        conv1 = tf.layers.conv2d(input, 16, 4, 2, activation=tf.nn.relu, name='conv1', reuse=reuse)
        conv2 = tf.layers.conv2d(conv1, 32, 3, 2, activation=tf.nn.relu, name='conv2', reuse=reuse)
        conv3 = tf.layers.conv2d(conv2, 64, 3, 2, activation=tf.nn.relu, name='conv3', reuse=reuse)
        conv4 = tf.layers.conv2d(conv3, 128, 3, 2, activation=tf.nn.relu, name='conv4', reuse=reuse)

        return conv4

    def __init__(self, split=False, normalize=True, fully_connected=None, learning_rate=1e-4, decay_steps=20000, decay_rate=0.5):
        self.graph = tf.Graph()

        with self.graph.as_default():

            self.batch_input = tf.placeholder(tf.float32, shape=[None, 64, 64, 2], name='inputs')
            self.batch_target = tf.placeholder(tf.float32, shape=[None, 2], name='targets')

            if normalize:
                with tf.name_scope('normalize'):
                    mean, var = tf.nn.moments(self.batch_input, axes=[1, 2], keep_dims=True)

                    head = tf.divide(tf.subtract(self.batch_input, mean), var)
            else:
                head = self.batch_input

            if split:
                split0, split1 = tf.split(head, 2, axis=3)

                conv0 = self.create_conv_layers(split0, reuse=False)
                flat0 = tools.layers.flatten(conv0)

                conv1 = self.create_conv_layers(split1, reuse=True)
                flat1 = tools.layers.flatten(conv1)

                head = tf.concat([flat0, flat1], 1)
            else:
                conv = self.create_conv_layers(head)
                head = tools.layers.flatten(conv)

            if fully_connected:
                head = tf.layers.dense(head, fully_connected, tf.nn.relu, name='FC')

            self.output = tf.layers.dense(inputs=head, units=2, name='output')

            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(tf.square(self.output - self.batch_target))

                self.summaries = {
                    'train': tf.summary.scalar('training', self.loss),
                    'val': tf.summary.scalar('validation', self.loss)
                }

            with tf.name_scope('train'):
                global_step = tf.Variable(0, trainable=False)

                lr_op = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, True)

                self.train_op = tf.train.AdamOptimizer(lr_op).minimize(self.loss, global_step)

                self.summaries['lr'] = tf.summary.scalar('learning_rate', lr_op)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

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

                    lr_summary = sess.run(self.summaries['lr'])

                    if verbose:
                        print('step: {} / loss: {:.4f} / val: {:.4f}'.format(step, train_loss, val_loss))

                    writer.add_summary(train_summary, step)
                    writer.add_summary(val_summary, step)
                    writer.add_summary(lr_summary, step)
                    writer.flush()

                if step % 1000 == 0:

                    self.saver.save(sess, log_path + 'model.ckpt')

    def predict(self, checkpoint_file, input_batch):

        with tf.Session(graph=self.graph) as sess:
            self.saver.restore(sess, checkpoint_file)

            output = sess.run(self.output, feed_dict={
                self.batch_input: input_batch
            })

        return output
