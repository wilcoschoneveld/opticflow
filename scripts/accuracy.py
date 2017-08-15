from model import CNN
from data.generator import DataGenerator
import tensorflow as tf

train_generator = DataGenerator(
    pattern='data/images/*/*',
    image_size=64,
    max_flow=5,
    max_scale=5,
    noise_level=5,
    interp='bicubic')

cnn = CNN(split=True, fully_connected=500, normalize=True, learning_rate=2e-4)

steps = []
losses = []
acclosses = []

with tf.Session(graph=cnn.graph) as sess:
    writer = tf.summary.FileWriter('.logs/accuracy-train-all/', graph=sess.graph)

    cnn.saver.restore(sess, '.logs/floyd/split/with-accuracy-saved/step42000.ckpt')

    with tf.variable_scope('accuracy2'):
        accuracy = tf.layers.dense(inputs=cnn.tmp, units=1, activation=lambda x: tf.exp(-0.25 * x))
        accuracy_loss = tf.reduce_mean(tf.squared_difference(accuracy, tf.exp(-0.25 * cnn.loss)))
        # acc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'accuracy2')
        train_acc_op = tf.train.AdamOptimizer(1e-4).minimize(accuracy_loss)
        acc_loss_summary = tf.summary.scalar('train', accuracy_loss)

    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'accuracy2'):
        sess.run(var.initializer)

    # train_acc_op = cnn.train_acc_op
    # accuracy_loss = cnn.accuracy_loss

    for step in range(1, 10000):

        inputs, targets = train_generator.generate_batch(batch_size=100)

        _, loss, acc_loss, train_sum, acc_sum = sess.run([train_acc_op, cnn.loss, accuracy_loss, cnn.summaries['train'], acc_loss_summary], feed_dict={
            cnn.batch_input: inputs,
            cnn.batch_target: targets
        })

        print('step: {} / loss: {:.4f} / acc_loss: {:.4f}'.format(step, loss, acc_loss))
        steps.append(step)
        losses.append(loss)
        acclosses.append(acc_loss)

        writer.add_summary(train_sum, step)
        writer.add_summary(acc_sum, step)

        if step % 1000 == 0:
            cnn.saver.save(sess, '.logs/output/step%i.ckpt' % step)

