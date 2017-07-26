import matplotlib.pyplot as plt
import cv2
import numpy as np

from data.generator import DataGenerator
from tests.test_generator import TestGenerator

gen = DataGenerator(
    pattern='data/images/test/city.jpg',
    image_size=64,
    max_flow=5,
    min_scale=3,
    max_scale=3,
    noise_level=0,
    sub_pixel_flow=True)

fast = cv2.FastFeatureDetector_create(threshold=80)

error = np.empty((9, 2))

for i in range(9):
    image0, image1, flow = gen.generate_flow()

    for threshold in np.arange(200, 5, -5):
        fast.setThreshold(threshold)

        kp = fast.detect(image0, None)

        if len(kp) > 40:
            break

    kp0 = cv2.KeyPoint_convert(kp)

    assert len(kp0)

    image0_kp = cv2.drawKeypoints(image0, kp, None, (255, 0, 0))

    kp1, st, err = cv2.calcOpticalFlowPyrLK(image0, image1, kp0, None, winSize=(10, 10), maxLevel=0)

    lk_flow = np.mean(np.fliplr(np.array(kp0) - np.array(kp1)), axis=0)

    error[i] = flow - lk_flow

    print('found %i points' % len(kp0))
    print('real flow:', flow)
    print('lk flow:', lk_flow)

    plt.subplot(3, 3, i + 1)
    TestGenerator._plot_flow(image0_kp, image1, flow, 64, 10)

print('MSE: ', np.mean(np.square(error)))

plt.show()
