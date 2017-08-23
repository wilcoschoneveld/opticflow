import datetime

from models.generator import DataGenerator
from models.cnn import CNN

train_generator = DataGenerator(
    pattern='data/train/*',
    image_size=64,
    max_flow=5,
    max_scale=5,
    noise_level=5,
    interp='bicubic')

validation_data = DataGenerator(
    pattern='data/test/*',
    image_size=64,
    max_flow=5,
    max_scale=5,
    noise_level=5,
    interp='bicubic').generate_batch(batch_size=100)

log_path = datetime.datetime.now().strftime('.logs/%Y%m%d-%H%M%S/')

cnn = CNN(split=True, fully_connected=500, normalize=True, learning_rate=2e-4)

cnn.train(train_generator, validation_data, 5000, log_path=log_path)
