from tensorflow.tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

from tools.filters import exp_average

acc = EventAccumulator('sun23july-1900-first_test.tfevents.ubuntuxps')
acc.Reload()

losses = acc.Scalars('loss/loss')
losses_x = [e.step for e in losses]
losses_y = exp_average(e.value for e in losses)

plt.plot(losses_x, losses_y)
plt.show()
