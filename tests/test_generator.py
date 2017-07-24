from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

from data.generator import DataGenerator


class TestGenerator(TestCase):

    @staticmethod
    def _plot_flow(image0, image1, flow, image_size, max_flow):
        plt.xlim(0, image_size + 2 * max_flow)
        plt.ylim(0, image_size + 2 * max_flow)

        a0 = max_flow
        b0 = max_flow + image_size

        a1 = a0 + flow[1]
        b1 = b0 + flow[1]
        a2 = a0 - flow[0]
        b2 = b0 - flow[0]

        plt.imshow(image0, extent=(a0, b0, a0, b0), cmap='gray', alpha=0.5)
        plt.imshow(image1, extent=(a1, b1, a2, b2), cmap='gray', alpha=0.5)

    def test_single_image(self):
        gen = DataGenerator('data/images/test/city.jpg', image_size=240, max_flow=50, max_scale=3, noise_level=20, interp='bicubic')

        image0, image1, flow = gen.generate_flow()

        self._plot_flow(image0, image1, flow, 240, 50)

        plt.show()

    def test_all_images(self):
        gen = DataGenerator('data/images/test/city.jpg', image_size=64, max_flow=10)

        for i in range(9):
            image0, image1, flow = gen.generate_flow()

            plt.subplot(3, 3, i + 1)
            self._plot_flow(image0, image1, flow, 64, 10)

        plt.show()

    def test_batch(self):
        gen = DataGenerator('data/images/train/*')

        inputs, targets = gen.generate_batch(100)

        self.assertEqual(inputs.shape, (100, 64, 64, 2))
        self.assertEqual(targets.shape, (100, 2))
