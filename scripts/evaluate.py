import numpy as np
import matplotlib.pyplot as plt

from data.generator import DataGenerator
from model import CNN
from tests.test_generator import TestGenerator
from tools.fastlk import FastLK

gen = DataGenerator(
    pattern='data/images/test/*',
    image_size=64,
    max_flow=5,
    min_scale=3,
    max_scale=3,
    noise_level=0,
    sub_pixel_flow=False)

# gen = DataGenerator(
#     pattern='data/images/*/*',
#     image_size=64,
#     max_flow=5,
#     max_scale=5,
#     noise_level=5,
#     interp='bicubic',
#     sub_pixel_flow=True)

inputs, targets = gen.generate_batch(1000)


# 0 prediction

error3 = np.mean(np.square(targets - 0), axis=1)


# CNN

cnn = CNN(split=False, normalize=True)

prediction = cnn.predict('.logs/floyd/small/longrun-saved/step80000.ckpt', inputs)

error = np.mean(np.square(prediction - targets), axis=1)


# CNN-split

cnn2 = CNN(split=True, normalize=True, fully_connected=500)

prediction = cnn2.predict('.logs/floyd/split/simple-saved/step44000.ckpt', inputs)

error4 = np.mean(np.square(prediction - targets), axis=1)


# FAST+LK

fastlk = FastLK(40, True)

flows = fastlk.batch_predict(inputs)

error2 = np.mean(np.square(targets - flows), axis=1)


# PLOT

plt.violinplot([error, error4, error2, error3], showmeans=True, showextrema=True, points=1000)
plt.ylim(-1, 26)
plt.gca().set_xticks([1, 2, 3, 4])
plt.gca().set_xticklabels(['CNN', 'CNN-split', 'FAST+LK', '0p'])
plt.ylabel('mean squared error (MSE)')

plt.figure()
plt.plot([1, 2], [3, 4])

for i in range(25):
    plt.subplot(5, 5, i + 1)

    image0 = inputs[i, :, :, 0]
    image1 = inputs[i, :, :, 1]
    flow = targets[i]

    TestGenerator._plot_flow(image0, image1, flow, gen.image_size, gen.max_flow)

plt.show()
