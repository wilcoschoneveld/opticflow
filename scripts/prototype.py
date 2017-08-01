from model import CNN
from data.generator import DataGenerator

train_generator = DataGenerator(
    pattern='data/images/train/garden.png',
    image_size=64,
    max_flow=5,
    max_scale=1,
    noise_level=0)

validation_data = DataGenerator(
    pattern='data/images/train/carpet.png',
    image_size=64,
    max_flow=5,
    max_scale=1,
    noise_level=0).generate_batch(batch_size=1000)

cnn = CNN(split=False, normalize=False, learning_rate=1e-4)

cnn.train(train_generator, validation_data, 25000)
