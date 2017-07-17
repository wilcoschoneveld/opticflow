import numpy
import tensorflow as tf


def flatten(inputs):
    shape = inputs.get_shape().as_list()
    size = numpy.prod(shape[1:])

    return tf.reshape(inputs, [-1, size])
