import numpy as np
import tensorflow as tf

from models.generator import DataGenerator
from models.cnn import CNN

gen = DataGenerator(
    pattern='data/test/*',
    image_size=64,
    max_flow=5,
    max_scale=5,
    noise_level=5,
    interp='bicubic',
    sub_pixel_flow=True)


data, targets = gen.generate_batch(1000)


cnn = CNN(split=True, fully_connected=500, normalize=True)


def find_variable(name):
    return [v for v in tf.global_variables() if v.name == name][0]


def prune_layer(name, amount):
    weights_var = find_variable(name)

    weights = sess.run(weights_var)

    to_prune = np.abs(weights) < amount*np.std(weights)

    weights[to_prune] = 0

    print('pruning %f percent of layer %s' % (100 * np.count_nonzero(to_prune) / weights.size, name))

    sess.run(weights_var.assign(weights))


with tf.Session(graph=cnn.graph) as sess:
    cnn.saver.restore(sess, 'checkpoints/split/step44000.ckpt')

    _, loss = sess.run([cnn.output, cnn.loss], feed_dict={cnn.batch_input: data, cnn.batch_target: targets})

    print('before pruning loss: ', loss)

    prune_layer('conv1/kernel:0', 0.2)
    prune_layer('conv2/kernel:0', 0.2)
    prune_layer('conv3/kernel:0', 0.2)
    prune_layer('conv4/kernel:0', 0.2)

    _, loss2 = sess.run([cnn.output, cnn.loss], feed_dict={cnn.batch_input: data, cnn.batch_target: targets})

    print('after pruning loss: ', loss2)
    print('difference (%): ', 100 * (loss2-loss) / loss)
