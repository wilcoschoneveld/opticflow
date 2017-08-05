import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from model import CNN


def find_variable(name):
    return [v for v in tf.global_variables() if v.name == name][0]


def plot_weights(name):
    weights = sess.run(find_variable(name))

    print(np.count_nonzero(np.abs(weights) < 0.01))
    print(np.count_nonzero(np.abs(weights) < 0.1*np.std(weights)))

    k = weights.shape[0]
    full = np.zeros((weights.shape[2]*(k+1), weights.shape[3]*(k+1)))

    for i in range(weights.shape[2]):
        for j in range(weights.shape[3]):
            full[i*(k+1):i*(k+1)+k, j*(k+1):j*(k+1)+k] = weights[:, :, i, j]

    full = np.flipud(full)
    v = 3 * np.std(weights)
    plt.imshow(full, cmap='PiYG', vmin=-v, vmax=v, extent=(0, weights.shape[3], 0, weights.shape[2]))

    # full = np.flipud(full) ** 2
    # plt.imshow(full, cmap='gray', extent=(0, weights.shape[3], 0, weights.shape[2]))

    plt.colorbar()
    plt.show()


cnn = CNN(split=False, normalize=True)
# cnn = CNN(split=True, fully_connected=500, normalize=True)

with tf.Session(graph=cnn.graph) as sess:
    cnn.saver.restore(sess, '.logs/floyd/small/longrun-saved/step80000.ckpt')
    # cnn.saver.restore(sess, '.logs/floyd/split/simple-saved/step44000.ckpt')

    for v in tf.trainable_variables():
        print(v)

    plot_weights('conv4/kernel:0')

    # weights = sess.run(find_variable('output/kernel:0'))
    #
    # plt.hist(weights, 100)
    # plt.show()