"""
    Class for a hawkes process.
"""
from typing import List
from dataclasses import dataclass

import numpy as np

from trajectory import Trajectory
from kernels import Kernel, State


@dataclass
class Hawkes:
    """
	A general multivariate Hawkes process that uses a sum of exponentials
        as the triggering kernel.

    """
    mus: np.ndarray
    W: np.ndarray #W[i][j] describes influence from event i to j
    kernel: Kernel #just a single triggering kernel, this is why a kernel shouldn't manage its state
    states: List[State] #each label gets its own state

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
            llh += segval

        return llh

    def calcsegllh(self, traj: Trajectory) -> float:
        """
            :param traj: The trajectory to calculate the log likelihood of
            :return: The log likelihood.
        """
        for label in traj.field["labels"]:
            self.states[label] = State()

        for time, delta, evlabel in traj:
            segscore = 0

            for label in traj.field["labels"]:
                self.states[label].advancestate(time)
                self.states[label].addevent(self.W[evlabel][label])

            for label in traj.field["labels"]:
                segscore += delta*self.states[label].intensities
                if evlabel != -1:
                    segscore -= np.log(self.states[label].intensities)

            yield segscore
