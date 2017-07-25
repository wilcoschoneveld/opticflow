from model import CNN
from data.generator import DataGenerator


class OverfitGenerator(object):
    @staticmethod
    def generate_batch(batch_size):
        return overfit_inputs, overfit_targets


overfit_inputs, overfit_targets = DataGenerator(
    pattern='data/images/train/*',
    image_size=64,
    max_flow=5,
    max_scale=5,
    noise_level=5,
    interp='bicubic').generate_batch(batch_size=100)

validation_data = DataGenerator(
    pattern='data/images/test/city.jpg',
    image_size=64,
    max_flow=5,
    max_scale=5,
    noise_level=5,
    interp='bicubic').generate_batch(batch_size=100)

cnn = CNN()

cnn.train(OverfitGenerator, validation_data, 1000)
