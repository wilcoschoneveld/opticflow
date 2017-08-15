import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from data.generator import DataGenerator
from model import CNN
from tests.test_generator import TestGenerator
from tools.fastlk import FastLK

gen = DataGenerator(
    pattern='data/images/*/*',
    image_size=64,
    max_flow=5,
    max_scale=5,
    noise_level=5,
    interp='bicubic',
    sub_pixel_flow=True)

inputs, targets = gen.generate_batch(1000)

cnn = CNN(split=True, fully_connected=500, normalize=True, learning_rate=2e-4)

with tf.Session(graph=cnn.graph) as sess:
    cnn.saver.restore(sess, '.logs/output/step4000.ckpt')

    prediction, accuracy = sess.run([cnn.output, cnn.accuracy], feed_dict={
            cnn.batch_input: inputs,
            cnn.batch_target: targets
    })

    error = np.mean(np.square(prediction - targets), axis=1)

    real_acc = np.exp(-0.25 * error)


plt.plot(accuracy, real_acc, '+')
plt.xlabel('predicted accuracy')
plt.ylabel('true accuracy')
plt.show()

plt.violinplot([real_acc, accuracy, accuracy - real_acc])
plt.gca().set_xticks([1, 2, 3])
plt.gca().set_xticklabels(['true', 'prediction', 'difference'])
plt.ylabel('accuracy')
plt.show()