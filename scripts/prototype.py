from models.generator import DataGenerator
from models.cnn import CNN

train_generator = DataGenerator(
    pattern='data/train/garden.png',
    image_size=64,
    max_flow=5,
    max_scale=1,
    noise_level=0)

validation_data = DataGenerator(
    pattern='data/train/carpet.png',
    image_size=64,
    max_flow=5,
    max_scale=1,
    noise_level=0).generate_batch(batch_size=1000)

convolution_layers = [
    [32, 3, 2],  # 32 filters, kernel size 3, stride 2
    [64, 3, 2],
    [128, 3, 1],
    [256, 3, 2]
]

cnn = CNN(config=convolution_layers, split=False, normalize=False, fully_connected=None, learning_rate=1e-4)

cnn.train(train_generator, validation_data, 25000)
