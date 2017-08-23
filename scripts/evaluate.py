import matplotlib.pyplot as plt
import numpy as np

from models.cnn import CNN
from models.generator import DataGenerator
from models.fastlk import FastLK

from tests.test_generator import TestGenerator

gen = DataGenerator(
    pattern='data/test/*',
    image_size=64,
    max_flow=5,
    min_scale=3,
    max_scale=3,
    noise_level=0,
    sub_pixel_flow=True)

# gen = DataGenerator(
#     pattern='data/*/*',
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

prediction = cnn.predict('checkpoints/normal/step81000.ckpt', inputs)

error = np.mean(np.square(prediction - targets), axis=1)


# CNN-split

cnn2 = CNN(split=True, normalize=True, fully_connected=500)

prediction = cnn2.predict('checkpoints/split/step44000.ckpt', inputs)

error4 = np.mean(np.square(prediction - targets), axis=1)


# FAST+LK

fastlk = FastLK(40, True)

flows = fastlk.batch_predict(inputs)

error2 = np.mean(np.square(targets - flows), axis=1)


# PLOT

print('CNN', np.mean(error))
print('split-CNN', np.mean(error4))
print('FAST+LK', np.mean(error2))
print('0p', np.mean(error3))

plt.violinplot([error, error4, error2, error3], showmeans=True, showextrema=True, points=1000)
plt.ylim(-0.5, gen.max_flow ** 2 + 1)
plt.gca().set_xticks([1, 2, 3, 4])
plt.gca().set_xticklabels(['CNN', 'split-CNN', 'FAST+LK', '0p'])
plt.ylabel('mean squared error (MSE)')

plt.figure()

for i in range(25):
    plt.subplot(5, 5, i + 1)

    image0 = inputs[i, :, :, 0]
    image1 = inputs[i, :, :, 1]
    flow = targets[i]

    TestGenerator._plot_flow(image0, image1, flow, gen.image_size, gen.max_flow)

plt.show()
