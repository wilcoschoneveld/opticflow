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

with tf.Session(graph=cnn.graph) as sess:

    cnn.saver.restore(sess, '.logs/output/step10.ckpt')

    for step in range(1, 1000):

        inputs, targets = train_generator.generate_batch(batch_size=100)

        _, loss = sess.run([cnn.train_acc_op, cnn.accuracy_loss], feed_dict={
            cnn.batch_input: inputs,
            cnn.batch_target: targets
        })

        print('step: {} / loss: {:.4f}'.format(step, loss))
