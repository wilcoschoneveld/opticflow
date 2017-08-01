from model import CNN
from data.generator import DataGenerator

train_generator = DataGenerator(
    pattern='/images/train/*',
    image_size=64,
    max_flow=5,
    max_scale=5,
    noise_level=5,
    interp='bicubic')

validation_data = DataGenerator(
    pattern='/images/test/*',
    image_size=64,
    max_flow=5,
    max_scale=5,
    noise_level=5,
    interp='bicubic').generate_batch(batch_size=100)

# log_pattern = datetime.datetime.now().strftime('.logs/%Y%m%d-%H%M%S/{}/')

cnn = CNN(split=False, normalize=True, learning_rate=1e-3)
# cnn = CNN(split=True, normalize=True, learning_rate=2e-4)

cnn.train(train_generator, validation_data, 100000, log_path='/output/')


# floyd run --cpu --env tensorflow-1.2 --data wilcoschoneveld/datasets/opticflow/1:images --tensorboard "python tuning.py"