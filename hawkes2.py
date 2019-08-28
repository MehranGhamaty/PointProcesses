
"""
    Class for a hawkre is process.
    Assumes that the session is executing eagerly.
"""
from typing import List
import tensorflow as tf
from trajectory import TimeSlice

class Hawkes:
    """
	A general multivariate Hawkes process that uses a sum of exponentials
        Non-modular kernel
    """
    def __init__(self, nlabels: int, betas: List[int]):
        self._nlabels: int = nlabels
        self._nkernels: int = len(betas)

        self._betas = tf.reshape(tf.Variable(
                initial_value=betas,
                dtype=tf.float32, trainable=True), (1, self._nkernels))

        self._weights = tf.Variable(
                initial_value=tf.random.uniform((self._nlabels, self._nlabels, self._nkernels),
                                                minval=0.001, maxval=1),
                dtype=tf.float32, trainable=True)

        self._mus = tf.Variable(
                initial_value=tf.random.uniform((self._nlabels,),
                                                minval=0.001, maxval=1),
                dtype=tf.float32, trainable=True)

        #each label and each kernel has its own state (used for tracking parents)
        self._states: tf.Tensor = tf.zeros((self._nlabels, self._nkernels))

    def __repr__(self):
        return "{}\n{}\n".format(self._weights, self._mus)

    def __call__(self, time_slice: TimeSlice) -> tf.Tensor:
        """
            Output is going to be the intensities
        """
        #decay the intensity
        self._states = tf.multiply(self._states, tf.exp(-time_slice.deltat * self._betas))
        #this adds to the intensities
        self._states = tf.cond(time_slice.label != -1, 
                lambda: self._states + self._weights[time_slice.label],
                lambda: self._states)
        return tf.add(self._mus, tf.reduce_sum(self._states,axis=1)) 

    @property
    def parameters(self):
        """ gets all the trainable parameters """
        return [self._weights, self._mus]

    def applygradients(self, eta, weightgrad, mugrad):
        """ applies updates with a step size """
        if weightgrad != None:
            tf.compat.v1.assign_add(self._weights, eta * weightgrad)
        if mugrad != None:
            tf.compat.v1.assign_add(self._mus, eta * mugrad)

    def calcsegllh(self, time_slice: TimeSlice) -> tf.Variable:
        """
            :param traj: The trajectory to calculate the log likelihood of
            :return: The log likelihood.
        """
    
        segscore = tf.Variable(tf.zeros((self._nlabels,)))
        intensities = self.__call__(time_slice)
        print("intensities are ", intensities)
        #return intensities #already lossing the gradients from the weights

        change = tf.multiply(time_slice.deltat, intensities)
        change = tf.cond(time_slice.label != -1,
                lambda: change - tf.math.log(intensities),
                lambda: change)
        segscore = tf.add(segscore, change)

        return tf.reduce_sum(segscore)
