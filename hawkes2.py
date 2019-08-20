
"""
    Class for a hawkre is process.
    Assumes that the session is executing eagerly.
"""
from typing import List
import tensorflow as tf
from trajectory import TimeSlice
from kernels import Kernel, State

class Hawkes:
    """
	A general multivariate Hawkes process that uses a sum of exponentials
    """
    def __init__(self, nlabels: int, kernel: Kernel):
        self._nlabels = nlabels
        self._kernel: Kernel = kernel
        self._weights = tf.Variable(
            initial_value=tf.random.uniform((nlabels, nlabels),
                                            minval=0.001, maxval=1),
            dtype=tf.float32, trainable=True)
        self._mus = tf.Variable(
            initial_value=tf.random.uniform((nlabels,),
                                            minval=0.001, maxval=1),
            dtype=tf.float32, trainable=True)
        self._states: tf.Tensor = tf.zeros((nlabels,))
        #self._states: List[State] = [State() for _ in range(nlabels)]

    def __repr__(self):
        return "{}\n{}\n".format(self._weights, self._mus)

    def __call__(self, time_slice: TimeSlice) -> tf.Tensor:
        """
            Output is going to be the intensities
        """
        #self._states = [self._kernel.advancestate(s, time_slice.time) for s in self._states]
        if time_slice.label != -1:
            self._states = [self._kernel.addevent(s, self._weights[time_slice.label][i])
                            for i, s in enumerate(self._states)]

        intensities = self._mus + tf.convert_to_tensor(
            [state.intensity for state in self._states], dtype=tf.float32)

        return intensities

    @property
    def parameters(self):
        """ gets all the trainable parameters"""
        return [self._weights, self._mus]

    def calcsegllh(self, time_slice: TimeSlice) -> tf.Variable:
        """
            :param traj: The trajectory to calculate the log likelihood of
            :return: The log likelihood.
        """
        segscore = tf.Variable(tf.zeros((self._nlabels,)))
        intensities = self.__call__(time_slice)
        return intensities #already lossing the gradients from the weights

        for label in range(self._nlabels):
            change = time_slice.deltat * (intensities[label])
            if time_slice.label != -1:
                change -= tf.math.log(intensities[label])
            segscore = segscore[label].assign(segscore[label] + change)

        return segscore
