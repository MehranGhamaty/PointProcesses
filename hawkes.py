"""
    Class for a hawkes process.
"""
from typing import List, Dict
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from trajectory import Trajectory
from kernels import Kernel, State

@dataclass
class Hawkes:
    """
	A general multivariate Hawkes process that uses a sum of exponentials
        as the triggering kernel.

    """
    def __init__(self, mus: tf.Tensor, W: tf.Tensor, kernel: Kernel):

        self._mus: tf.Tensor = mus
        self._weights: tf.Tensor = W
        self._kernel: Kernel = kernel
        #this should be initialized per label
        self._mus: tf.
        self._states: List[State] = [State() for _ in range(W.shape[0])]

    def sample(self, end_time) -> Trajectory:
        """
	    :param end_time: The end time of the sample
	    :return: The sampled trajectory
	"""
        pass

    def discretellh(self, traj: Trajectory) -> float:
        """
            :param traj: The trajectory to calculate the llh of
        """
        llh: float = 0.
        for segval in self.calcsegllh(traj):
            for score in segval.values():
                llh += score

        return llh

    def calcsegllh(self, traj: Trajectory) -> Dict[str, float]:
        """
            :param traj: The trajectory to calculate the log likelihood of
            :return: The log likelihood.
        """
        for label in traj.field["labels"]:
            self._states[label] = State()

        for time, delta, evlabel in traj:
            segscore = {}

            for label in traj.field["labels"]:
                self._states[label].advancestate(time)
                self._states[label].addevent(self._weights[evlabel][label])
                segscore[label] = 0

            for label in traj.field["labels"]:
                segscore[label] += delta*self._states[label].intensities
                if evlabel != -1:
                    segscore[label] -= np.log(self._states[label].intensities)

            yield segscore
