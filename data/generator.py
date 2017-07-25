import glob
import numpy as np
import scipy.misc
import random


class DataGenerator(object):

    def __init__(self, pattern='images/*', image_size=64, max_flow=10, min_scale=1, max_scale=5, noise_level=5, interp='bicubic'):
        self.image_size = image_size
        self.max_flow = max_flow
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.noise_level = noise_level
        self.interp = interp

        self.images = [scipy.misc.imread(f, flatten=True) for f in glob.glob(pattern)]

        if not self.images:
            raise FileNotFoundError("No images match pattern '{}'".format(pattern))

    def generate_flow(self):
        image = random.choice(self.images)

        scale = np.random.randint(self.min_scale, self.max_scale + 1)
        flow = np.random.randint(-self.max_flow*scale, self.max_flow*scale, size=2)

        y0 = np.random.randint(self.max_flow*scale, image.shape[0] - self.image_size*scale - self.max_flow*scale)
        x0 = np.random.randint(self.max_flow*scale, image.shape[1] - self.image_size*scale - self.max_flow*scale)

        y1 = y0 + flow[0]
        x1 = x0 + flow[1]

        image0 = image[y0:y0+self.image_size*scale, x0:x0+self.image_size*scale]
        image1 = image[y1:y1+self.image_size*scale, x1:x1+self.image_size*scale]

        if scale > 1:
            image0 = scipy.misc.imresize(image0, (self.image_size, self.image_size), self.interp)
            image1 = scipy.misc.imresize(image1, (self.image_size, self.image_size), self.interp)

        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, (2, self.image_size, self.image_size))

            image0 = np.clip(image0 + noise[0], 0, 255).round().astype(np.uint8)
            image1 = np.clip(image1 + noise[1], 0, 255).round().astype(np.uint8)

        return image0, image1, flow / scale

    def generate_batch(self, batch_size):
        inputs = np.empty((batch_size, self.image_size, self.image_size, 2))
        targets = np.empty((batch_size, 2))

        for i in range(batch_size):
            image0, image1, flow = self.generate_flow()

            inputs[i, :, :, 0] = image0
            inputs[i, :, :, 1] = image1

            targets[i] = flow

        return inputs, targets
