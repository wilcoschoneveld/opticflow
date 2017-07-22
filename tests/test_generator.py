from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

from data.generator import DataGenerator


class TestGenerator(TestCase):

    @staticmethod
    def _match_images(image0, image1, flow, image_size, max_flow):
        canvas = np.zeros((image_size + 2*max_flow, image_size + 2*max_flow))

        a0 = max_flow
        b0 = max_flow + image_size

        a1 = a0 + flow[0]
        b1 = b0 + flow[0]
        a2 = a0 + flow[1]
        b2 = b0 + flow[1]

        canvas[a0:b0, a0:b0] += image0 * 0.5
        canvas[a1:b1, a2:b2] += image1 * 0.5

        return canvas

    def test_single_image(self):
        gen = DataGenerator('data/images/train/garden.jpg', image_size=240, max_flow=50, max_scale=10, noise_level=20)

        image0, image1, flow = gen.generate_flow()
        combined = self._match_images(image0, image1, flow, 240, 50)

        plt.imshow(combined, cmap='gray')
        plt.show()

    def test_all_images(self):
        gen = DataGenerator('data/images/train/*', image_size=64, max_flow=10)

        for i in range(25):
            image0, image1, flow = gen.generate_flow()
            combined = self._match_images(image0, image1, flow, 64, 10)

            plt.subplot(5, 5, i + 1)
            plt.imshow(combined, cmap='gray')

        plt.show()

    def test_batch(self):
        gen = DataGenerator('data/images/train/*')

        inputs, targets = gen.generate_batch(100)

        self.assertEqual(inputs.shape, (100, 64, 64, 2))
        self.assertEqual(targets.shape, (100, 2))
