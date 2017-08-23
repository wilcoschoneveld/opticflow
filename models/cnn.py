import tensorflow as tf

import tools


class CNN(object):
    """A Convolutional Neural Network architecture for predicting optical flow.
    """

    def __init__(self, config=None, split=False, normalize=True, fully_connected=None, learning_rate=1e-4):
        """Creates a new architecture.

        Args:
          config: Optional. A list of convolution layer configurations, where each config is in
            the form (num filter, kernel size, stride).
          split: If true, builds the architecture as two tunnels with shared weights. The input
            images are split and processed separately. See the report for more info.
          normalize: Normalize the input by subtracting the mean and dividing by the variance.
          fully_connected: Optional. Adds a dense layer before the output with the given number
            of units.
          learning_rate: Initial learning rate used for the Adam optimizer.
        """
        # Create a new graph and store the reference
        self.graph = tf.Graph()

        # Create a mapping for holding summary objects
        self.summaries = {}

        # User-provided configuration or a default setup of 4 convolution layers
        self.config = config or [
            [16, 4, 2],     # 16 filters, kernel size 4, stride 2
            [32, 3, 2],
            [64, 3, 2],
            [128, 3, 2]
        ]

        # Create a context where all operations are added to the right graph.
        with self.graph.as_default():

            self.batch_input = tf.placeholder(tf.float32, shape=[None, 64, 64, 2], name='inputs')
            self.batch_target = tf.placeholder(tf.float32, shape=[None, 2], name='targets')

            # Start building with the input placeholder
            head = self.batch_input

            # Add input preprocessor if requested
            if normalize:
                head = self._create_normalize(head)

            if split:
                # Split the input into two separate images
                split0, split1 = tf.split(head, 2, axis=3)

                # Create a convolutional tunnel and flatten the result
                conv0 = self._create_conv_layers(split0, reuse=False)
                flat0 = tools.layers.flatten(conv0)

                # Create another convolutional tunnel and reuse weights
                conv1 = self._create_conv_layers(split1, reuse=True)
                flat1 = tools.layers.flatten(conv1)

                # Concatenate the output
                head = tf.concat([flat0, flat1], 1)
            else:
                # Create a convolutional tunnel for the input
                conv = self._create_conv_layers(head)

                # Flatten the last convolution layer
                head = tools.layers.flatten(conv)

            # Add a fully connected hidden layer if requested
            if fully_connected:
                head = tf.layers.dense(head, fully_connected, tf.nn.relu, name='FC')

            # Create the prediction layer
            self.output = tf.layers.dense(inputs=head, units=2, name='output')

            with tf.name_scope('loss'):
                # Optical flow loss is mean squared error
                self.loss = tf.reduce_mean(tf.squared_difference(self.output, self.batch_target))

                # Create and save summaries for training and validation loss
                self.summaries['train'] = tf.summary.scalar('training', self.loss)
                self.summaries['val'] = tf.summary.scalar('validation', self.loss)

            with tf.name_scope('train'):
                # Global step counter used for learning rate decay
                global_step = tf.Variable(0, trainable=False)

                # Exponential decay of 0.5 at 20k steps
                lr_op = tf.train.exponential_decay(learning_rate, global_step, 20000, 0.5, staircase=True)

                # Create Adam optimizer and minimize for the mean squared error loss
                self.train_op = tf.train.AdamOptimizer(lr_op).minimize(self.loss, global_step)

                # Create tensorboard summary for learning rate value
                self.summaries['lr'] = tf.summary.scalar('learning_rate', lr_op)

            # Create global variables initializer and saver
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

    def _create_normalize(self, head):
        with tf.name_scope('normalize'):
            # Calculate mean and variance for each image separately
            mean, var = tf.nn.moments(head, axes=[1, 2], keep_dims=True)

            # Subtract mean and divide by variance to normalize the input
            return tf.divide(tf.subtract(head, mean), var)

    def _create_conv_layers(self, head, reuse=False):
        for i, (filters, kernel_size, strides) in enumerate(self.config):
            # Stack a convolution layer with given configuration and the ReLU activation function
            head = tf.layers.conv2d(
                inputs=head,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                activation=tf.nn.relu,
                name='conv' + str(i + 1),
                reuse=reuse)

        # Return output of last convolution layer
        return head

    def train(self, train_generator, validation_data, max_steps, verbose=True, log_path=".logs/latest/"):
        """Train the architecture with a dataset generator to predict optical flow.

        Args:
          train_generator: An instance of a DataGenerator which can provide batches of image pairs and target flows.
          validation_data: A single batch (images, targets) to track model performance.
          max_steps: Maximum number of steps the network will train for, this value is equal to the number of batches
            that will be extracted from the DataGenerator.
          verbose: Print the training and validation loss to the standard output
          log_path: Folder where to save checkpoints and tensorboard summaries.
        """
        # Launch a session with the network architecture as graph
        with tf.Session(graph=self.graph) as sess:
            # Create a summary writer and store the graph
            writer = tf.summary.FileWriter(log_path, graph=sess.graph)

            # Initialize all global variables (network weights)
            sess.run(self.init)

            # Save the initialized weights
            self.saver.save(sess, log_path + 'init.ckpt')

            # Start with the first step of the training loop
            for step in range(1, max_steps + 1):
                # Generate a batch of 100 image pairs and target flow values
                inputs, targets = train_generator.generate_batch(batch_size=100)

                # Run the optimizer with generated batch
                sess.run(self.train_op, feed_dict={
                    self.batch_input: inputs,
                    self.batch_target: targets
                })

                # Evaluate the network training and validation loss every 25 steps
                if step % 25 == 0:
                    # Obtain the training loss with the generated batch
                    train_summary, train_loss = sess.run([self.summaries['train'], self.loss], feed_dict={
                        self.batch_input: inputs,
                        self.batch_target: targets
                    })

                    # Obtain the validation loss with the provided validation data
                    val_summary, val_loss = sess.run([self.summaries['val'], self.loss], feed_dict={
                        self.batch_input: validation_data[0],
                        self.batch_target: validation_data[1]
                    })

                    # Obtain the current learning rate
                    lr_summary = sess.run(self.summaries['lr'])

                    # Print the current step with training and validation loss
                    if verbose:
                        print('step: {} / loss: {:.4f} / val: {:.4f}'.format(step, train_loss, val_loss))

                    # Write the summaries to the log file
                    writer.add_summary(train_summary, step)
                    writer.add_summary(val_summary, step)
                    writer.add_summary(lr_summary, step)
                    writer.flush()

                # Save a checkpoint of the architecture every 1000 steps with the trained weight values
                if step % 1000 == 0:
                    self.saver.save(sess, log_path + 'step%i.ckpt' % step)

    def predict(self, checkpoint_file, input_batch):
        """Predict optical flow for a given batch of image pairs.

        Args:
          checkpoint_file: Trained weights to use when evaluating the network.
          input_batch: A batch of image pairs for which to predict optical flow.
        """
        with tf.Session(graph=self.graph) as sess:
            # Restore the trained weights into the session
            self.saver.restore(sess, checkpoint_file)

            # Run the network with the given input batch and retrieve output predictions
            output = sess.run(self.output, feed_dict={
                self.batch_input: input_batch
            })

        return output
