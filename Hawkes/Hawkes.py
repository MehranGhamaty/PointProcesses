
"""
    Author: Mehran Ghamaty

    Class for a hawkre is process.
    Assumes that there is a tensorflow session executing eagerly.


    Do not try to sample and perform gradient descent.
"""
from typing import List
from numpy import inf

import tensorflow as tf

from Trajectory.Trajectory import Trajectory
from Trajectory.TimeSlice import TimeSlice
from Trajectory.Field import Field

class Hawkes:
    """
       A general multivariate Hawkes process that uses a sum of exponentials
       Non-modular kernel. Some methods manage state, be aware.

       Assumes labels are from the space [0, nlabels)
    """
    def __init__(self, nlabels: int, betas: List[int]):
        self._nlabels: int = nlabels
        self._nkernels: int = len(betas)

        self._betas = tf.reshape(tf.Variable(
            initial_value=betas,
            dtype=tf.float32, trainable=True), (1, self._nkernels))

        # I need better ways of initializing these....
        # mu should make sure everthing is always positive....
        # constraints..... I really feel like using the lagrangian would be better than other
        # other methods I've seen.....

        self._weights: tf.Variable = tf.Variable(
            initial_value=tf.random.uniform((self._nlabels, self._nlabels, self._nkernels),
                                            minval=0.001, maxval=1),
            dtype=tf.float32, trainable=True)

        self._mus: tf.Variable = tf.Variable(
            initial_value=tf.random.uniform((self._nlabels,),
                                            minval=0.001, maxval=1),
            dtype=tf.float32, trainable=True)

        # each label and each kernel has its own state (used for tracking parents)
        self._states: tf.Tensor = tf.zeros((self._nlabels, self._nkernels))
        self._currtime: float = 0  # used in sampling

    def __repr__(self):
        return "{}\n{}\n".format(self._weights, self._mus)


    def __call__(self, time_slice: TimeSlice) -> tf.Tensor:
        """
            Output is going to be the intensities
        """
        # decay the intensity
        self._states = tf.multiply(self._states,
                                   tf.exp(-time_slice.deltat * self._betas))

        # this adds to the intensities
        self._states = tf.cond(time_slice.label != -1,
                               lambda: self._states +
                               self._weights[time_slice.label],
                               lambda: self._states)
        # print("state values ", self._states)
        # self._states = tf.nn.relu(self._states)  # remove negative values
        return tf.add(self._mus, tf.reduce_sum(self._states, axis=1))

    def getsupp(self) -> tf.Tensor:
        """
            For sampling we generate the supp of the intensity.
            Doesn't change the actual state, it just returns
            the total intensity as if an event of each time occured
        """
        return tf.reduce_sum(self._mus) + \
               tf.reduce_sum(self._states) + \
               tf.reduce_sum(self._weights)
        # I'm not a 100% on that


    def resetstate(self):
        """ sets the state to 0 """
        self._states: tf.Tensor = tf.zeros((self._nlabels, self._nkernels))
        self._currtime = 0

    @property
    def parameters(self):
        """ gets all the trainable parameters """
        return [self._weights, self._mus]

    @property
    def state(self):
        """ returns the state of each variable """
        return tf.add(self._mus, tf.reduce_sum(self._states, axis=1))

    def setparams(self, newweights, newmu):
        """ assigns new values """
        tf.compat.v1.assign(self._weights, newweights)
        tf.compat.v1.assign(self._mus, newmu)

    def applygradients(self, eta, weightgrad, mugrad):
        """ applies updates with a step size """
        tf.compat.v1.assign_sub(self._weights, eta * weightgrad)
        tf.compat.v1.assign_sub(self._mus, eta * mugrad)

        tf.compat.v1.assign(self._weights, tf.nn.relu(self._weights))
        tf.compat.v1.assign(self._mus, tf.nn.relu(self._mus))

    def calcsegnegllh(self, intensities, time_slice: TimeSlice) -> tf.Variable:
        """
            This isn't using self, and is sepecific to sum of
            exponentials. The intensities would have to match
            the kernel.

            :param intensities: the current state of intensities
                (make sure base rates are included)
            :param traj: The trajectory to calculate the log likelihood of
            :return: The log likelihood.
        """
        volume = tf.multiply(time_slice.deltat, intensities)
        segscore = tf.cond(time_slice.label != -1,
                           lambda: volume - tf.math.log(intensities),
                           lambda: volume)
        return tf.reduce_sum(segscore)

    def calcllh(self, traj: Trajectory) -> tf.Variable:
        """
            Calculates the log likelihood over entire trajectory
            Note: this handles resetting the state
        """
        llh = 0
        self.resetstate()
        for time_slice in traj:
            ints = self(time_slice)
            llh -= self.calcsegnegllh(ints, time_slice)
        self.resetstate()
        return llh

    def gradstep(self, time_slice: TimeSlice, eta: float) -> tf.Variable:
        """ takes a single step from a time slice """
        with tf.GradientTape() as tape:
            ints = self(time_slice)
            llh = self.calcsegnegllh(ints, time_slice)
        gradients = tape.gradient(llh, self.parameters)
        self.applygradients(eta, gradients[0], gradients[1])
        return llh

    def gradient_descent(self, traj: Trajectory, eta: float) -> tf.Variable:
        """ Performs gradient ascent in order to fit parameters """
        for time_slice in traj:
            yield self.gradstep(time_slice, eta)

    def gradient_descent_full(self, traj: Trajectory, eta: float) -> tf.Variable:
        """ Performs gradient descent over trajectory, handles internal state """
        llh = 0
        self.resetstate()
        for negllh in self.gradient_descent(traj, eta):
            llh -= negllh
        self.resetstate()
        return llh

    def sample_next_event(self) -> TimeSlice:
        """
            given that the current state is correct, sample the next
            TimeSlice
            First we get the time until next event;
        """
        while True:
            lambdabar = self.getsupp()
            to_inverse = tf.random.uniform([1],
                                           minval=0, maxval=1,
                                           dtype=tf.float32)

            deltat = - tf.math.log(to_inverse) / lambdabar
            self._currtime += deltat
            intensities = self(TimeSlice(time=self._currtime, deltat=deltat,
                                         label=-1))

            totalint = tf.reduce_sum(intensities)
            reject = tf.random.uniform([1],
                                       minval=0, maxval=1,
                                       dtype=tf.float32)

            if reject*lambdabar <= totalint:
                # I need to use superposition to get the label
                # to sample then I need to spin through the items
                labelrng = tf.random.uniform([1], minval=0, maxval=totalint)
                sellabel = -1
                while labelrng > 0:
                    sellabel += 1
                    labelrng -= intensities[sellabel]
                mylab = tf.constant(sellabel, dtype=tf.int32)
                yield TimeSlice(time=self._currtime, deltat=deltat, label=mylab)

    def sample(self, max_time: float, tau: float = inf) -> Trajectory:
        """
            Generates a trajectory of from time 0 to max_time
            Stateless function
        """
        traj = Trajectory({"times" : Field(values=[], continuous=True, space=(0, max_time)),
                           "labels" : Field(values=[], continuous=False,
                                            space=tf.convert_to_tensor(
                                                [i for i in range(self._nlabels)]))},
                          tau=tau)
        self.resetstate()
        for time_slice in self.sample_next_event():
            if time_slice.time > max_time:
                break
            traj.add_time_slice(time_slice)
        self.resetstate()

        return traj
