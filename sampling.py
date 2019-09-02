"""

A sampling test

"""

import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, MovieWriter
import numpy as np

tf.enable_eager_execution()

from trajectory import Field, Trajectory, TimeSlice
from hawkes import Hawkes

# make our dummy dataset
LABELSET = tf.convert_to_tensor([0, 1], dtype=tf.int32)
TIMES = tf.convert_to_tensor([1., 2.5, 5, 9.], dtype=tf.float32)
LABELS = tf.convert_to_tensor([0, 1, 0, 1], dtype=tf.int32)
FIELDS = {
    "times": Field(values=TIMES, continuous=True, space=(0., 10.)),
    "labels": Field(values=LABELS, continuous=False, space=LABELSET)
}
TRAJ = Trajectory(FIELDS, tau=0.1)

# construct our dummy hawkes process, with three exponential kernels
HP = Hawkes(len(LABELSET), [1, 3, 4])

for i, llh in enumerate(HP.gradientascent(TRAJ, 0.1)):
    if i > len(TRAJ):
        break

times = []
labels = []
for timeslice in HP.samplenextevent():
    times.append(timeslice.time.numpy()[0])

    labels.append(timeslice.label.numpy())

    if len(times) > 50:
        break
print(times)
print(labels)
