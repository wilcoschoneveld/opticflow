

# early stopping conditions
#       xplosion
#       time
#       max epoch


# overfit
# try very low learning rate (find min)
# try very high learning rate (find max)
# hyperparameter tuning
# hyperparameter tuning 2 (finer search)
# more..?

# model function
# inputs=architecture, preprocess data
#
# train function
# inputs=batch generator, validation generator, max_steps, batch_size, learning_rate, regularization, verbose=True

from model import CNN
from data.generator import DataGenerator

validation_data = DataGenerator(
    pattern='data/images/test/city.jpg',
    image_size=64,
    max_flow=5,
    max_scale=5,
    noise_level=5,
    interp='bicubic').generate_batch(batch_size=100)

train_generator = DataGenerator(
    pattern='data/images/train/*',
    image_size=64,
    max_flow=5,
    max_scale=5,
    noise_level=5,
    interp='bicubic')


overfit_inputs, overfit_targets = train_generator.generate_batch(batch_size=100)

class OverfitGenerator(object):
    @staticmethod
    def generate_batch(batch_size=100):
        return overfit_inputs, overfit_targets

cnn = CNN()

cnn.train(OverfitGenerator, validation_data, 25000, True)