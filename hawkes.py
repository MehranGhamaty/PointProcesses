
"""
    Class for a hawkre is process.
    Assumes that the session is executing eagerly.
"""
from typing import List
import tensorflow as tf
from trajectory import TimeSlice, Trajectory


class Hawkes:
    """
       A general multivariate Hawkes process that uses a sum of exponentials
       Non-modular kernel. Having internal state might be problematic
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

        self._weights = tf.Variable(
            initial_value=tf.random.uniform((self._nlabels, self._nlabels, self._nkernels),
                                            minval=0.001, maxval=1),
            dtype=tf.float32, trainable=True)

        self._mus = tf.Variable(
            initial_value=tf.random.uniform((self._nlabels,),
                                            minval=0.001, maxval=1),
            dtype=tf.float32, trainable=True)

        # each label and each kernel has its own state (used for tracking parents)
        self._states: tf.Tensor = tf.zeros((self._nlabels, self._nkernels))

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

    @property
    def parameters(self):
        """ gets all the trainable parameters """
        return [self._weights, self._mus]

    def applygradients(self, eta, weightgrad, mugrad):
        """ applies updates with a step size """
        tf.compat.v1.assign_sub(self._weights, eta * weightgrad)
        tf.compat.v1.assign_sub(self._mus, eta * mugrad)

    def calcsegnegllh(self, intensities, time_slice: TimeSlice) -> tf.Variable:
        """
            :param intensities: the current state of intensities
            :param traj: The trajectory to calculate the log likelihood of
            :return: The log likelihood.
        """
        segscore = tf.Variable(tf.zeros((self._nlabels,)))
        # intensities = self.__call__(time_slice)
        # print("intensities are ", intensities)

        change = tf.multiply(time_slice.deltat, intensities)
        change = tf.cond(time_slice.label != -1,
                         lambda: change - tf.math.log(intensities),
                         lambda: change)
        segscore = tf.add(segscore, change)
        return tf.reduce_sum(segscore)

    def gradstep(self, time_slice: TimeSlice, eta: float):
        """ """

    def gradientdescent(self, traj: Trajectory, eta: float):
        """ Performs gradient ascent in order to fit parameters """
        for time_slice in traj:
            with tf.GradientTape() as tape:
                ints = self(time_slice)
                # Add the new intensities
                # print(ints)
                llh = self.calcsegnegllh(ints, time_slice)
            gradients = tape.gradient(llh, self.parameters)
            self.applygradients(eta, gradients[0], gradients[1])
            yield llh

    def samplenextevent(self):
        """
            given that the current state is correct, sample the next
            TimeSlice
            First we get the time until next event;
        """
        currtime = 0  # time to keep track of supp
        while True:
            lambdabar = self.getsupp()
            to_inverse = tf.random_uniform([1],
                                           minval=0, maxval=1,
                                           dtype=tf.float32)

            deltat = - tf.math.log(to_inverse / lambdabar)
            currtime += deltat
            intensities = self(TimeSlice(time=currtime, deltat=deltat,
                                         label=-1))

            totalint = tf.reduce_sum(intensities)
            reject = tf.random_uniform([1],
                                       minval=0, maxval=1,
                                       dtype=tf.float32)
            if reject*lambdabar <= totalint:
                # I need to use superposition to get the label
                # to sample then I need to spin through the items
                labelrng = tf.random_uniform([1], minval=0, maxval=totalint)
                sellabel = -1
                while labelrng > 0:
                    sellabel += 1
                    labelrng -= intensities[sellabel]
                mylab = tf.constant(sellabel, dtype=tf.int32)
                yield TimeSlice(time=currtime, deltat=deltat, label=mylab)
