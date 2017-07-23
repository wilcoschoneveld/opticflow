import tensorflow as tf
import matplotlib.pyplot as plt

from tools.filters import exp_average

losses_x = []
losses_y = []

for e in tf.train.summary_iterator('sun23july-1900-first_test.tfevents.ubuntuxps'):

    for v in e.summary.value:
        if v.tag == "loss/loss":
            losses_x.append(e.step)
            losses_y.append(v.simple_value)

plt.plot(losses_x, losses_y, alpha=0.3)
plt.plot(losses_x, exp_average(losses_y, 0.8))
plt.show()
