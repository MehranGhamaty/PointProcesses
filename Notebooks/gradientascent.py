"""

A test using GA

"""

#Import our libraries
import tensorflow as tf
import matplotlib.pyplot as plt

tf.enable_eager_execution()

from trajectory import Field, Trajectory
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
# Perform GA
eta = 0.5
scores = []
change = []

# track the intensities, how am I going to plot this exactly?
# I need two points per time(one with the previous int and the new one)
intensities0 = []
intensities1 = []
times = []

while len(scores) < 10:
    totalllh = 0
    ints0 = []
    ints1 = []
    time = []
    HP.resetstate()
    for time_slice in TRAJ:
        with tf.GradientTape() as tape:
            ints = HP(time_slice)
            # Add the old ints to get a stepwise constant
            if len(ints0) > 0:
                ints0.append(ints0[-1])
                ints1.append(ints1[-1])
                time.append(time_slice.time.numpy())
            # Add the new intensities
            ints0.append(ints.numpy()[0])
            ints1.append(ints.numpy()[1])
            time.append(time_slice.time.numpy())
            # print(ints)
            llh = HP.calcsegnegllh(ints, time_slice)
        gradients = tape.gradient(llh, HP.parameters)
        #print(gradients)
        totalllh += llh
        HP.applygradients(eta, gradients[0], gradients[1])
    # print(totalllh)
    intensities0.append(ints0)
    intensities1.append(ints1)
    times.append(time)
    scores.append(totalllh.numpy())

x = []
y = []
for time_slice in TRAJ:
    lab = time_slice.label.numpy()
    if lab != -1:
        x.append(time_slice.time.numpy())
        y.append(lab)

for it, time in enumerate(times):
    plt.close()
    plt.plot(time, intensities0[it])
    plt.plot(time, intensities1[it])
    plt.scatter(x, [v*1 for v in y])
    plt.pause(0.5)


plt.plot(time, intensities0[-1])
plt.plot(time, intensities1[-1])
plt.scatter(x, [v*1 for v in y])
plt.xlabel("Time")
plt.ylabel("Intensities")
plt.title("Example fit sampled data with two labels")
plt.savefig('intensities.pdf')
