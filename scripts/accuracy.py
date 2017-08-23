import tensorflow as tf

from models.generator import DataGenerator
from models.cnn import CNN


def accuracy_function(x):
    return tf.exp(-0.25 * x)


train_generator = DataGenerator(
    pattern='data/*/*',
    image_size=64,
    max_flow=5,
    max_scale=5,
    noise_level=5,
    interp='bicubic')

cnn = CNN(split=True, fully_connected=500, normalize=True, learning_rate=2e-4)

RETRAIN_ALL = False

with tf.Session(graph=cnn.graph) as sess:
    writer = tf.summary.FileWriter('.logs/accuracy', graph=sess.graph)

    cnn.saver.restore(sess, 'checkpoints/split/step44000.ckpt')

    head = cnn.graph.get_tensor_by_name('FC/Relu:0')

    with tf.variable_scope('accuracy'):
        accuracy = tf.layers.dense(inputs=head, units=1, activation=accuracy_function)
        accuracy_loss = tf.reduce_mean(tf.squared_difference(accuracy, accuracy_function(cnn.loss)))

        if not RETRAIN_ALL:
            acc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'accuracy')
            train_acc_op = tf.train.AdamOptimizer(1e-4).minimize(accuracy_loss, var_list=acc_vars)
        else:
            train_acc_op = tf.train.AdamOptimizer(1e-4).minimize(accuracy_loss)

        acc_loss_summary = tf.summary.scalar('train', accuracy_loss)

    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'accuracy'):
        sess.run(var.initializer)

    for step in range(1, 10000):

        inputs, targets = train_generator.generate_batch(batch_size=100)

        evaluate = [train_acc_op, cnn.loss, accuracy_loss, cnn.summaries['train'], acc_loss_summary]

        _, loss, acc_loss, train_sum, acc_sum = sess.run(evaluate, feed_dict={
            cnn.batch_input: inputs,
            cnn.batch_target: targets
        })

        print('step: {} / loss: {:.4f} / acc_loss: {:.4f}'.format(step, loss, acc_loss))

        writer.add_summary(train_sum, step)
        writer.add_summary(acc_sum, step)

        if step % 1000 == 0:
            cnn.saver.save(sess, '.logs/accuracy/step%i.ckpt' % step)

