import tensorflow as tf
import matplotlib.pyplot as plt


def filter_values(values, weight):
    last = values[0]
    smoothed = []

    for value in values:
        last = last * weight + value * (1 - weight)
        smoothed.append(last)

    return smoothed


losses_x = []
losses_y = []

for e in tf.train.summary_iterator('sun23july-1900-first_test.ubuntuxps'):

    for v in e.summary.value:
        if v.tag == "loss/loss":
            losses_x.append(e.step)
            losses_y.append(v.simple_value)

plt.plot(losses_x, losses_y, alpha=0.3)
plt.plot(losses_x, filter_values(losses_y, 0.8))
plt.show()
