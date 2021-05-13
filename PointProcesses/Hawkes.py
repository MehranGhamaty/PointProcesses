"""
    Author: Mehran Ghamaty

    Class for a Hawkes process.
    Assumes that there is a tensorflow session executing eagerly.

    Do not try to sample and perform gradient descent at the same time. 
    (You can if you know what your doing).
    
"""
from typing import List
from numpy import inf

import tensorflow as tf

from PointProcesses.PointProcess import PointProcess

from DataStores.Trajectory import Trajectory, TimeSlice

class Hawkes(PointProcess):
    """
       A general multivariate Hawkes process that uses a sum of exponentials
       non-modular kernel. Methods manage state, be aware.

       Works a lot better if mus and betas are approximately correct,
       use the set params function to initialize them reasonably.

       This can be done with the set params method, but
       initialization method that takes a sample trajectory should be written.

       Each set of nlabels gets its own matrix, this is a good way
       of representing mutiple unconnected attributes.

       Assumes labels are from the space [0, nlabels)
    """

    def __init__(self, nlabels: List[int]):
        # super(Hawkes, self).__init__(nlabels)
        self.__nkernels: int = len(nlabels)

        # self.__betas = tf.reshape(tf.Variable(
        #    initial_value=betas,
        #    dtype=tf.float32, trainable=True), (1, self.__nkernels))

        self.__betas: tf.Variable = tf.Variable(
            initial_value=tf.random.uniform((1, self.__nkernels))
        )

        # I need better ways of initializing these....
        # mu should make sure everthing is always above zero....
        # constraints..... using the lagrangian would be
        # better than other other methods I've seen.....
        self.__weights: tf.Variable = tf.Variable(
            initial_value=tf.random.uniform((self.__nlabels,
                                             self.__nlabels, self.__nkernels),
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

    def check_add(self, states: tf.Tensor, time_slice: TimeSlice) -> tf.Tensor:
        """
            Returns a tensor that represents the new state
        """
        states = tf.multiply(states, tf.exp(-time_slice.deltat * self.__betas))
        states = tf.cond(time_slice.label != -1,
                         lambda: self.__states + self.__weights[time_slice.label],
                         lambda: self.__states)
        return states

    def getsupp(self) -> tf.Tensor:
        """
            For sampling we generate the supp of the intensity.
            Doesn't change the actual state, it just returns
            the total intensity as if an event of each time occured
        """
        return tf.reduce_sum(self.__mus) + \
               tf.reduce_sum(self.__states) + \
               tf.reduce_sum(self.__weights)

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

    @property
    def stable(self):
        """ returns true if spectral radius is less than 1 """
        return True

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

    @staticmethod
    def calcsegnegllh(intensities, time_slice: TimeSlice) -> tf.Variable:
        """
            Calculates the segments negative log likelihood
            Since its the sum of exponentials I can do this exactly....
            Just sume the integrals

            :param intensities: the current state of intensities
            :param traj: The trajectory to calculate the log likelihood of
            :return: The log likelihood.
        """
        #this is actually a pretty terrible approximation
        # volume = tf.multiply(time_slice, intensities)
        volume = 1 - tf.math.exp(-intensities * time_slice.deltat)
        segscore = tf.cond(time_slice.label != -1,
                           lambda: volume - tf.math.log(intensities),
                           lambda: volume)
        return tf.reduce_sum(segscore)

    def calcllh(self, traj: Trajectory) -> tf.Variable:
        """
            Calculates the log likelihood over entire trajectory
        """
        self.resetstate()
        llh = self.calcllh(traj)
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
        states: tf.Tensor = tf.zeros((self.__nlabels, self.__nkernels))

        while True:
            lambdabar = self.getsupp()
            to_inverse = tf.random.uniform([1],
                                           minval=0, maxval=1,
                                           dtype=tf.float32)

            deltat = - tf.math.log(to_inverse) / lambdabar
            self.__currtime += deltat

            tmp_states = self.check_add(states, TimeSlice(time=self.__currtime, deltat=deltat,
                                                          label=-1))

            intensities = tf.add(self.__mus, tf.reduce_sum(tmp_states, axis=1))

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
                time_slice = TimeSlice(time=self.__currtime, deltat=deltat, label=mylab)
                states = self.check_add(states, time_slice)
                yield time_slice

    def sample(self, max_time: float, tau: float = inf) -> Trajectory:
        """
            Generates a trajectory of from time 0 to max_time
            Stateless function
        """
        self.resetstate()
        traj = super(Hawkes, self).sample(max_time, tau)
        self.resetstate()
        return traj
