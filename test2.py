"""
    Small test for sgd
"""

import tensorflow as tf
import matplotlib.pyplot as plt

tf.compat.v1.enable_eager_execution()

from trajectory import Field, Trajectory
from hawkes import Hawkes

LABELSET = tf.convert_to_tensor([0, 1], dtype=tf.int32)

TIMES = tf.convert_to_tensor([1., 2., 6., 8.], dtype=tf.float32)
LABELS = tf.convert_to_tensor([0, 1, 0, 1], dtype=tf.int32)
FIELDS = {
    "times" : Field(values=TIMES, continuous=True, space=(0., 10.)),
    "labels" : Field(values=LABELS, continuous=False, space=LABELSET)
}
TRAJ = Trajectory(FIELDS, tau=1.)
print(TRAJ)


HP = Hawkes(len(LABELSET), [1, 3, 4] )
eta = 0.1

score = []


for epoch in range(2):
    totalllh = 0
    print(epoch, " has params ", HP)
    for time_slice in TRAJ:
        print("time slice ", time_slice)
        with tf.GradientTape() as tape:
            ints = HP(time_slice)
            llh = HP.calcsegllh(ints, time_slice)
        gradients = tape.gradient(llh, HP.parameters)
        print("gradients", gradients)
        print("log likelihood is ", llh)
        totalllh += llh
        HP.applygradients(eta, gradients[0], gradients[1]) 

        #it seems like 

    score.append(totalllh)
    print(score)
exit(1)

print("DONE LEARNING")
print(score)


#print(HP)
