import numpy as np
import matplotlib.pyplot as plt

from data.generator import DataGenerator
from model import CNN
from tools.fastlk import FastLK

gen = DataGenerator(
    pattern='data/images/test/*',
    image_size=64,
    max_flow=5,
    min_scale=3,
    max_scale=3,
    noise_level=0,
    normalize=True,
    sub_pixel_flow=False)

# gen = DataGenerator(
#     pattern='data/images/*/*',
#     image_size=64,
#     max_flow=5,
#     max_scale=5,
#     noise_level=20,
#     normalize=True,
#     interp='bicubic',
#     sub_pixel_flow=True)

inputs, targets = gen.generate_batch(1000)

cnn = CNN(split=False, learning_rate=1e-3)

prediction = cnn.predict('/home/wilco/Documents/test/www.floydhub.com/viewer/data/MaZYxftMMsqdQNzaf9LnwT/RAtTj8Q9G6thsHK8E4L5Ba/model.ckpt', inputs)

error = np.mean(np.square(prediction - targets), axis=1)

prediction2 = cnn.predict('/home/wilco/Documents/test/test3/www.floydhub.com/viewer/data/MaZYxftMMsqdQNzaf9LnwT/RAtTj8Q9G6thsHK8E4L5Ba/model.ckpt', inputs)

error2 = np.mean(np.square(prediction - targets), axis=1)

error4 = np.mean(np.square(targets - 0), axis=1)

gen.normalize = False

inputs, targets = gen.generate_batch(1000)

fastlk = FastLK(40, True)

flows = fastlk.batch_predict(inputs)

print(len(flows))

error3 = np.mean(np.square(targets - flows), axis=1)

plt.violinplot([error, error2, error3, error4], showmeans=True, showextrema=True)
plt.ylim(-1, 26)
plt.gca().set_xticks([1, 2, 3, 4])
plt.gca().set_xticklabels(['cnn1', 'cnn2', 'flk', '0p'])

plt.show()