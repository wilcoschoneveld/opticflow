import glob
import scipy.ndimage
import numpy as np
import random


class DataGenerator(object):

    def __init__(self, pattern='images/*', image_size=64, max_flow=10, max_scale=5, noise_level=5):
        self.image_size = image_size
        self.max_flow = max_flow
        self.max_scale = max_scale
        self.noise_level = noise_level

        self.images = [scipy.ndimage.imread(f, flatten=True) for f in glob.glob(pattern)]

    def generate_flow(self):
        image = random.choice(self.images)

        scale = np.random.randint(1, self.max_scale)
        noise = np.random.normal(0, self.noise_level, (2, self.image_size, self.image_size))
        flow = np.random.randint(-self.max_flow, self.max_flow, size=2)

        y0 = np.random.randint(self.max_flow*scale, image.shape[0] - self.image_size*scale - self.max_flow*scale)
        x0 = np.random.randint(self.max_flow*scale, image.shape[1] - self.image_size*scale - self.max_flow*scale)

        y1 = y0 + flow[0] * scale
        x1 = x0 + flow[1] * scale

        image0 = image[y0:y0+self.image_size*scale:scale, x0:x0+self.image_size*scale:scale]
        image1 = image[y1:y1+self.image_size*scale:scale, x1:x1+self.image_size*scale:scale]

        image0 = np.clip(image0 + noise[0], 0, 255).round().astype(np.uint8)
        image1 = np.clip(image1 + noise[1], 0, 255).round().astype(np.uint8)

        return image0, image1, flow

    def generate_batch(self, batch_size):
        inputs = np.empty((batch_size, self.image_size, self.image_size, 2))
        targets = np.empty((batch_size, 2))

        for i in range(batch_size):
            image0, image1, flow = self.generate_flow()

            inputs[i, :, :, 0] = image0
            inputs[i, :, :, 1] = image1

            targets[i] = flow

        return inputs, targets
