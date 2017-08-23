from models.generator import DataGenerator
from models.cnn import CNN


class OverfitGenerator(object):
    @staticmethod
    def generate_batch(batch_size):
        return overfit_inputs, overfit_targets


overfit_inputs, overfit_targets = DataGenerator(
    pattern='data/train/*',
    image_size=64,
    max_flow=5,
    max_scale=5,
    noise_level=5,
    interp='bicubic').generate_batch(batch_size=100)

validation_data = DataGenerator(
    pattern='data/test/*',
    image_size=64,
    max_flow=5,
    max_scale=5,
    noise_level=5,
    interp='bicubic').generate_batch(batch_size=100)

cnn = CNN(split=False, normalize=True, fully_connected=None, learning_rate=1e-4)

cnn.train(OverfitGenerator, validation_data, 1000)
