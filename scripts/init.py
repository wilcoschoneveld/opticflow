import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from model import CNN


def find_variable(name):
    return [v for v in tf.global_variables() if v.name == name][0]


cnn = CNN(split=True, fully_connected=500, normalize=True)

with tf.Session(graph=cnn.graph) as sess:
    for i in range(1, 5):

        cnn.saver.restore(sess, '.logs/floyd/split/init/init.ckpt')
        weights0 = sess.run(find_variable('conv%i/kernel:0' % i))

        cnn.saver.restore(sess, '.logs/floyd/split/init/step2000.ckpt')
        weights1 = sess.run(find_variable('conv%i/kernel:0' % i))

        cnn.saver.restore(sess, '.logs/floyd/split/init/step3000.ckpt')
        weights2 = sess.run(find_variable('conv%i/kernel:0' % i))

        cnn.saver.restore(sess, '.logs/floyd/split/init/step5000.ckpt')
        weights3 = sess.run(find_variable('conv%i/kernel:0' % i))

        plt.hist([weights0, weights1, weights3], 20,
                 label=['initialized', 'step 2000', 'step 3000'], histtype='step')

        plt.xlabel('weight value')
        plt.ylabel('frequency')
        plt.legend()

        plt.show()

    # weights = sess.run(find_variable('output/kernel:0'))
    #
    # plt.hist(weights, 100)
    # plt.show()