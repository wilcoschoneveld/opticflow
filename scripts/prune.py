import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from data.generator import DataGenerator
from model import CNN


gen = DataGenerator(
    pattern='data/images/test/*',
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
    cnn.saver.restore(sess, '.logs/floyd/split/with-accuracy-saved/step42000.ckpt')

    _, loss = sess.run([cnn.output, cnn.loss], feed_dict={cnn.batch_input: data, cnn.batch_target: targets})

    print(loss)

    prune_layer('conv1/kernel:0', 0.2)
    prune_layer('conv2/kernel:0', 0.2)
    prune_layer('conv3/kernel:0', 0.2)
    prune_layer('conv4/kernel:0', 0.2)

    _, loss2 = sess.run([cnn.output, cnn.loss], feed_dict={cnn.batch_input: data, cnn.batch_target: targets})

    print(loss2)
    print(100*loss2/loss)
