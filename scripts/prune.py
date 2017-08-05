import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from data.generator import DataGenerator
from model import CNN


gen = DataGenerator(
    pattern='data/images/*/*',
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

with tf.Session(graph=cnn.graph) as sess:
    cnn.saver.restore(sess, '.logs/floyd/split/simple-saved/step44000.ckpt')

    prediction, loss = sess.run([cnn.output, cnn.loss], feed_dict={cnn.batch_input: data, cnn.batch_target: targets})

    print(loss)

    weights_var = find_variable('conv3/kernel:0')

    weights = sess.run(weights_var)

    to_prune = np.abs(weights) < 0.3*np.std(weights)

    weights[to_prune] = 0

    print('pruning %f percent' % (100 * np.count_nonzero(to_prune) / weights.size))

    sess.run(weights_var.assign(weights))

    prediction, loss = sess.run([cnn.output, cnn.loss], feed_dict={cnn.batch_input: data, cnn.batch_target: targets})

    print(loss)
