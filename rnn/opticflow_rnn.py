from tensorflow.contrib.keras.python import keras
import data

NUM_STEPS = 4       # Note: keep it short when using unroll=True
NUM_CHANNELS = 1
WINDOW_SIZE = 64


model = keras.models.Sequential()

model.add(
    keras.layers.TimeDistributed(
        keras.layers.Conv2D(
            filters=64,
            kernel_size=(5, 5),
            strides=2,
            activation='relu'
        ),
        input_shape=(NUM_STEPS, WINDOW_SIZE, WINDOW_SIZE, NUM_CHANNELS)
    )
)

model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(
    filters=128,
    kernel_size=(3, 3),
    strides=2,
    activation='relu'
)))

model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(
    filters=128,
    kernel_size=(3, 3),
    strides=2,
    activation='relu'
)))

model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(
    filters=256,
    kernel_size=(3, 3),
    strides=2,
    activation='relu'
)))

model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))

# model.add(keras.layers.SimpleRNN(
#     units=128,
#     activation='tanh',
#     unroll=True,
#     return_sequences=True
#
#     # initialize state to zero??
# ))

model.add(keras.layers.LSTM(
    units=512,
    activation='relu',
    unroll=True,
    return_sequences=True
))

model.add(keras.layers.TimeDistributed(keras.layers.Dense(2)))

model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-4),
    loss='mean_squared_error',
)

model.summary()

checkpoints = keras.callbacks.ModelCheckpoint('tmp/log2/weights.{epoch:02d}-{loss:.2f}.h5')
tensorboard = keras.callbacks.TensorBoard('tmp/log2')

model.fit_generator(data.sequence_batch_generator(20, NUM_STEPS, WINDOW_SIZE),
                    steps_per_epoch=1000,
                    epochs=20,
                    callbacks=[checkpoints, tensorboard])
