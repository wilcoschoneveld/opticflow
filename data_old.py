import scipy.ndimage
import numpy

MAX_MOVE = 5   # pixels

full_img = scipy.ndimage.imread('data/images/train/garden.jpg', flatten=True)   # (height=4032, width=3024)


def generate_sample(window_size=240):
    y = numpy.random.randint(MAX_MOVE, full_img.shape[0] - window_size - MAX_MOVE)
    x = numpy.random.randint(MAX_MOVE, full_img.shape[1] - window_size - MAX_MOVE)

    flow = numpy.random.random_integers(-MAX_MOVE, MAX_MOVE, size=2)

    img0 = full_img[y:y+window_size, x:x+window_size]
    img1 = full_img[y+flow[0]:y+window_size+flow[0], x+flow[1]:x+window_size+flow[1]]

    return img0, img1, flow


def generate_batches(batch_size=10, window_size=240):

    batch_inputs = numpy.zeros((batch_size, window_size, window_size, 2))
    batch_targets = numpy.zeros((batch_size, 2))

    while True:
        for i in range(batch_size):
            img0, img1, flow = generate_sample(window_size)

            batch_inputs[i, :, :, 0] = img0
            batch_inputs[i, :, :, 1] = img1

            batch_targets[i, :] = flow

        yield batch_inputs, batch_targets


def sequence_generator(seq_len=5, window_size=240):

    margin = MAX_MOVE*seq_len

    y = numpy.random.randint(margin, full_img.shape[0] - window_size - margin)
    x = numpy.random.randint(margin, full_img.shape[1] - window_size - margin)
    move_y = 0
    move_x = 0

    inputs = numpy.zeros((seq_len, window_size, window_size, 1))
    outputs = numpy.zeros((seq_len, 2))

    for i in range(seq_len):
        inputs[i, :, :, 0] = full_img[y:y+window_size, x:x+window_size]
        outputs[i, :] = (move_y, move_x)

        move_y = numpy.random.randint(-MAX_MOVE, MAX_MOVE)
        move_x = numpy.random.randint(-MAX_MOVE, MAX_MOVE)

        y += move_y
        x += move_x

    return inputs, outputs


def sequence_batch_generator(batch_size=10, seq_len=5, window_size=240):
    batch_inputs = numpy.zeros((batch_size, seq_len, window_size, window_size, 1))
    batch_outputs = numpy.zeros((batch_size, seq_len, 2))

    # infinitely generate batches
    while True:
        for i in range(batch_size):

            inputs, outputs = sequence_generator(seq_len, window_size)

            batch_inputs[i] = inputs
            batch_outputs[i] = outputs

        yield batch_inputs, batch_outputs


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    _batch_size = 3
    _seq_len = 7
    _size = 240

    batch_inputs, batch_outputs = sequence_batch_generator(_batch_size, _seq_len, _size).__next__()

    f, axarr = plt.subplots(_batch_size, _seq_len)

    for j in range(_batch_size):
        for i in range(_seq_len):
            axarr[j, i].imshow(batch_inputs[j, i, :, :, 0], cmap="gray")
            axarr[j, i].set_title(batch_outputs[j, i])

    plt.show()
