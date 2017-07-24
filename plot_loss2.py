from tensorflow.tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

from tools.filters import exp_average

acc = EventAccumulator('tmp/log/events.out.tfevents.1500878665.ubuntuxps')
acc.Reload()

training_x = [e.step for e in acc.Scalars('loss/training')]
training_y = exp_average(e.value for e in acc.Scalars('loss/training'))

validation_x = [e.step for e in acc.Scalars('loss/validation')]
validation_y = exp_average(e.value for e in acc.Scalars('loss/validation'))

plt.plot(training_x, training_y, label='training')
plt.plot(validation_x, validation_y, label='validation')
plt.legend()
plt.show()
