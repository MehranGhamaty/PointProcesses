"""
    A script to test the features of the Hawkes process
"""
import tensorflow as tf
import matplotlib.pyplot as plt

tf.enable_eager_execution()

from trajectory import Field, Trajectory
from hawkes import Hawkes

#make our dummy dataset
LABELSET = tf.convert_to_tensor([0, 1], dtype=tf.int32)
TIMES = tf.convert_to_tensor([1., 2.5, 5, 9.], dtype=tf.float32)
LABELS = tf.convert_to_tensor([0, 1, 0, 1], dtype=tf.int32)
FIELDS = {
    "times" : Field(values=TIMES, continuous=True, space=(0., 10.)),
    "labels" : Field(values=LABELS, continuous=False, space=LABELSET)
}
TRAJ = Trajectory(FIELDS, tau=0.1)
# construct our dummy hawkes process, with three exponential kernels
HP = Hawkes(len(LABELSET), [1, 3, 4])
# Perform GA
ETA = 100
SCORES = []

# track the INTENSITIES, how am I going to plot this exactly?
# I need two poINTS per TIME(one with the previous int and the new one)
INTENSITIES0 = []
INTENSITIES1 = []
TIMES = []

while len(SCORES) < 10:
    TOTALLLH = 0
    INTS0 = []
    INTS1 = []
    TIME = []
    HP.resetstate()
    for time_slice in TRAJ:
        with tf.GradientTape() as tape:
            INTS = HP(time_slice)
            #Add the old INTS to get a stepwise constant
            if not INTS0:
                INTS0.append(INTS0[-1])
                INTS1.append(INTS1[-1])
                TIME.append(time_slice.TIME.numpy())
            #Add the new INTENSITIES
            INTS0.append(INTS.numpy()[0])
            INTS1.append(INTS.numpy()[1])
            TIME.append(time_slice.TIME.numpy())
            #print(INTS)
            llh = HP.calcsegllh(INTS, time_slice)
        gradients = tape.gradient(llh, HP.parameters)
        TOTALLLH += llh
        HP.applygradients(ETA, gradients[0], gradients[1])
    #print(TOTALLLH)
    INTENSITIES0.append(INTS0)
    INTENSITIES1.append(INTS1)
    TIMES.append(TIME)
    SCORES.append(TOTALLLH.numpy())

X = []
Y = []
for time_slice in TRAJ:
    l = time_slice.label.numpy()
    if l != -1:
        X.append(time_slice.time.numpy())
        Y.append(l)
for i, time in enumerate(TIMES):
    plt.close()
    plt.plot(time, INTENSITIES0[i])
    plt.plot(time, INTENSITIES1[i])
    plt.scatter(X, [v*3000 for v in Y])
    plt.pause(0.5)
    plt.close()
