"""
    Author: Mehran Ghamaty

    Class for a hawkre is process.
    Assumes that there is a tensorflow session executing eagerly.


    Do not try to sample and perform gradient descent.
"""
from typing import List
from numpy import inf

import tensorflow as tf

from PointProcesses.PointProcess import PointProcess

from DataStores.Trajectory import Trajectory, TimeSlice, Field

class Hawkes(PointProcess):
    """
       A general multivariate Hawkes process that uses a sum of exponentials
       Non-modular kernel. Some methods manage state, be aware.

       Assumes labels are from the space [0, nlabels)
    """
    def __init__(self, nlabels: int, betas: List[int]):
        self.__nlabels: int = nlabels
        self.__nkernels: int = len(betas)

        self.__betas = tf.reshape(tf.Variable(
            initial_value=betas,
            dtype=tf.float32, trainable=True), (1, self.__nkernels))

        # I need better ways of initializing these....
        # mu should make sure everthing is always positive....
        # constraints..... I really feel like using the lagrangian would be better than other
        # other methods I've seen.....

        self.__weights: tf.Variable = tf.Variable(
            initial_value=tf.random.uniform((self.__nlabels, self.__nlabels, self.__nkernels),
                                            minval=0.001, maxval=1),
            dtype=tf.float32, trainable=True)

        self.__mus: tf.Variable = tf.Variable(
            initial_value=tf.random.uniform((self.__nlabels,),
                                            minval=0.001, maxval=1),
            dtype=tf.float32, trainable=True)

        # each label and each kernel has its own state (used for tracking parents)
        self.__states: tf.Tensor = tf.zeros((self.__nlabels, self.__nkernels))
        self.__currtime: float = 0  # used in sampling

    def __repr__(self):
        return "{}\n{}\n".format(self.__weights, self.__mus)


    def __call__(self, time_slice: TimeSlice) -> tf.Tensor:
        """
            Output is going to be the intensities
        """
        # decay the intensity
        self.__states = tf.multiply(self.__states,
                                   tf.exp(-time_slice.deltat * self.__betas))

        # this adds to the intensities
        self.__states = tf.cond(time_slice.label != -1,
                               lambda: self.__states +
                               self.__weights[time_slice.label],
                               lambda: self.__states)
        # print("state values ", self.__states)
        # self.__states = tf.nn.relu(self.__states)  # remove negative values
        return tf.add(self.__mus, tf.reduce_sum(self.__states, axis=1))

    def getsupp(self) -> tf.Tensor:
        """
            For sampling we generate the supp of the intensity.
            Doesn't change the actual state, it just returns
            the total intensity as if an event of each time occured
        """
        return tf.reduce_sum(self.__mus) + \
               tf.reduce_sum(self.__states) + \
               tf.reduce_sum(self.__weights)
        # I'm not a 100% on that


    def resetstate(self):
        """ sets the state to 0 """
        self.__states: tf.Tensor = tf.zeros((self.__nlabels, self.__nkernels))
        self.__currtime = 0

    @property
    def parameters(self):
        """ gets all the trainable parameters """
        return [self.__weights, self.__mus]

    @property
    def state(self):
        """ returns the state of each variable """
        return tf.add(self.__mus, tf.reduce_sum(self.__states, axis=1))

    def setparams(self, newweights, newmu):
        """ assigns new values """
        tf.compat.v1.assign(self.__weights, newweights)
        tf.compat.v1.assign(self.__mus, newmu)

    def applygradients(self, eta, weightgrad, mugrad):
        """ applies updates with a step size """
        tf.compat.v1.assign_sub(self.__weights, eta * weightgrad)
        tf.compat.v1.assign_sub(self.__mus, eta * mugrad)

        tf.compat.v1.assign(self.__weights, tf.nn.relu(self.__weights))
        tf.compat.v1.assign(self.__mus, tf.nn.relu(self.__mus))

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
            self.__currtime += deltat
            intensities = self(TimeSlice(time=self.__currtime, deltat=deltat,
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
                yield TimeSlice(time=self.__currtime, deltat=deltat, label=mylab)

    def sample(self, max_time: float, tau: float = inf) -> Trajectory:
        """
            Generates a trajectory of from time 0 to max_time
            Stateless function
        """
        traj = Trajectory({"times" : Field(values=[], continuous=True, space=(0, max_time)),
                           "labels" : Field(values=[], continuous=False,
                                            space=tf.convert_to_tensor(
                                                [i for i in range(self.__nlabels)]))},
                          tau=tau)
        self.resetstate()
        for time_slice in self.sample_next_event():
            if time_slice.time > max_time:
                break
            traj.add_time_slice(time_slice)
        self.resetstate()

        return traj