import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tools.filters import exp_average


def plot_loss(event_file):
    losses_x = []
    losses_y = []

    for e in tf.train.summary_iterator(event_file):

        for v in e.summary.value:
            if v.tag == "loss/training":
                losses_x.append(e.step)
                losses_y.append(v.simple_value)

    plt.plot(losses_x, losses_y, 'C0', alpha=0.3, label='training')
    plt.plot(losses_x, exp_average(losses_y, 0.9), 'C0', label='filtered')
    plt.legend()
    plt.show()


def plot_loss2(event_file):
    acc = EventAccumulator(event_file)
    acc.Reload()

    training_x = [e.step for e in acc.Scalars('loss/training')]
    training_y = exp_average(e.value for e in acc.Scalars('loss/training'))

    validation_x = [e.step for e in acc.Scalars('loss/validation')]
    validation_y = exp_average(e.value for e in acc.Scalars('loss/validation'))

    plt.plot(training_x, training_y, label='training')
    plt.plot(validation_x, validation_y, label='validation')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_loss('checkpoints/normal/events.out.tfevents.1501583753.task-instance-container')
    plot_loss2('checkpoints/split/events.out.tfevents.1501586174.task-instance-container')
