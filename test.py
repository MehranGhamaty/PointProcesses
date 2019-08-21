"""
    Small test for sgd
"""

import tensorflow as tf

tf.compat.v1.enable_eager_execution()

from trajectory import Field, Trajectory
from kernels import ExponentialKernel
from hawkes import Hawkes

LABELS = tf.convert_to_tensor([0, 1], dtype=tf.int32)

TIMES = tf.convert_to_tensor([1., 2., 6., 8.], dtype=tf.float32)
LABELS = tf.convert_to_tensor([0, 1, 0, 1], dtype=tf.int32)
FIELDS = {
    "times" : Field(values=TIMES, continuous=True, space=(0., 10.)),
    "labels" : Field(values=LABELS, continuous=False, space=LABELS)
}
TRAJ = Trajectory(FIELDS, tau=1.)
print(TRAJ)

HP = Hawkes(len(LABELS), ExponentialKernel())

print(HP)
HP.sgd(0.1, TRAJ)
print(HP)