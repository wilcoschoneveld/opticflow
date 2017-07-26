import matplotlib.pyplot as plt

from data.generator import DataGenerator

train_generator = DataGenerator(
    pattern='data/images/train/*',
    image_size=64,
    max_flow=5,
    max_scale=5,
    noise_level=5,
    interp='bicubic')

inputs, _ = train_generator.generate_batch(100)

plt.subplot(1, 2, 1)
plt.hist(inputs.flatten(), 128)
plt.xlim(0, 255)

train_generator.normalize = True

norms, _ = train_generator.generate_batch(100)

plt.subplot(1, 2, 2)
plt.hist(norms.flatten(), 128)
plt.xlim(-1, 1)
plt.show()
