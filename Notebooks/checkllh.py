"""
Attempting to relearn the parameters.
"""
from importlib import reload
import tensorflow as tf
import matplotlib.pyplot as plt
import copy
import numpy as np

tf.enable_eager_execution()
tf.random.set_random_seed(0)
import trajectory as tr
import hawkes as h

reload(tr)
reload(h)


# make our dummy dataset
LABELSET = tf.convert_to_tensor([0, 1], dtype=tf.int32)
# construct our dummy hawkes process, with three exponential kernels
NLABELS = len(LABELSET)
KERNELPARAMS = [1]
HP = h.Hawkes(NLABELS, KERNELPARAMS)
HP.setparams(
tf.random.uniform((NLABELS, NLABELS, len(KERNELPARAMS)),
                                minval=10, maxval=100),
    tf.random.uniform((NLABELS,), minval=10, maxval=20))
print(HP.parameters)
HP2 = h.Hawkes(len(LABELSET), KERNELPARAMS)
ETA = 0.1
ENDTIMES = 1


TRAJ = tr.Trajectory(
    {
        "times" : tr.Field(values=[], continuous=True, space=(0, ENDTIMES)),
        "labels" : tr.Field(values=[], continuous=False, space=LABELSET)
    }, 0.2)


HPLLH = 0
HP2LLH = 0

HPLLHS = []
HP2LLHS = []

L0TIMES = []
L1TIMES = []

DISTS = []
TIMES = []


HP.resetstate()
HP2.resetstate()
for timeslice in HP.samplenextevent():
    time = timeslice.time.numpy()[0]
    if time > ENDTIMES:
        break

    if timeslice.label.numpy() == 0:
        L0TIMES.append(time)
    else:
        L1TIMES.append(time)
    HP2.gradstep(timeslice, ETA)
    TIMES.append(timeslice.time.numpy()[0])

    # I want actual LLH
    HPLLH -= HP.calcsegnegllh(HP.state, timeslice).numpy()
    HPLLHS.append(HPLLH)

    wdist = tf.norm(HP.parameters[0] - HP2.parameters[0])
    mdist = tf.norm(HP.parameters[1] - HP2.parameters[1])
    DISTS.append((wdist + mdist).numpy())

    # I want LLH after taking the gradient step...
    HP2LLH -= HP2.calcsegnegllh(HP2.state, timeslice).numpy()
    HP2LLHS.append(HP2LLH)
    print(HPLLH, HP2LLH)
    plt.plot(TIMES, HPLLHS, label="Expert LLH")
    plt.plot(TIMES, HP2LLHS, label="Learner LLH")
    plt.plot(TIMES, DISTS, label="L2 Distance for parameters")

    plt.scatter(L0TIMES, [0 for _ in L0TIMES], color='red', label='type 0')
    plt.scatter(L1TIMES, [0 for _ in L1TIMES], color='blue', label='type 1')
    plt.legend()
    plt.xlabel("Time: {} events".format(len(TIMES)))
    plt.ylabel("Log Likelihoods")
    plt.title("Relearning from a Sampled Model")
    plt.show()
    plt.pause(0.01)

plt.plot(TIMES, HPLLHS, label="Expert LLH")
plt.plot(TIMES, HP2LLHS, label="Learner LLH")
plt.plot(TIMES, DISTS, label="L2 Distance for parameters")
plt.scatter(L0TIMES, [0 for _ in L0TIMES], color='red', label='type 0')
plt.scatter(L1TIMES, [0 for _ in L1TIMES], color='blue', label='type 1')
plt.legend()
plt.xlabel("Time: {} events".format(len(TIMES)))
plt.ylabel("Log Likelihoods")
plt.title("Relearning from a Sampled Model")
plt.show()
plt.pause(0.01)
plt.savefig("metris.pdf")
