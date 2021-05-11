"""
    This is a class for superposition of Poisson Process.
    
"""

import numpy as np
import tensorflow as tf

from PointProcesses.PointProcess import PointProcess
from Trajectory.Trajectory import Trajectory, TimeSlice, Field

class PoissonProcess(PointProcess):
    """
        A superposition of Poisson Processes
        written for online learning.

        No prior (the advantage of online learning comes from large datasets
        where the prior won't really help at all anyway). Actually it depends on where 
        your definition of prior comes from.
    """

    def __init__(self, nlabels: int, max_memory: float = np.inf):
        super(PoissonProcess, self).__init__(nlabels, max_memory)
        self.__intensities = tf.zeros((nlabels,), dtype=tf.dtypes.float32)
        self.__num_events = tf.zeros((nlabels,), dtype=tf.dtypes.float32)

        self.__total_time = 0
        self.__eventqueue = []

        # For use with sampling
        self.__currtime = 0 # assuming stationarity....

    def sample_next_event(self) -> TimeSlice:
        """
            samples an event from the process using superposition.
            parameters are not re-estimated while sampling. 
        """
        lambdabar = np.sum(self.__intensities)
        while True:

            to_inverse = np.random.random()
            deltat = - np.log(to_inverse) / lambdabar
            self.__currtime += deltat

            #now find the label of the next event
            labelrng = np.random.random()*lambdabar
            sellabel = -1
            while labelrng > 0:
                sellabel += 1
                labelrng -= self.__intensities[sellabel]
            mylab = tf.constant(sellabel, dtype=tf.int32)
            yield TimeSlice(time=self.__currtime, deltat=deltat, label=mylab)

    def __call__(self, time_slice: TimeSlice) -> tf.Tensor:
        return self.add_time_slice(time_slice)

    def resetstate(self):
        """
            Resets the internal state of the intensities
        """
        nlabels = super(PoissonProcess, self).nlabels
        self.__intensities = tf.zeros((nlabels,), dtype=tf.dtypes.float32)
        self.__num_events = tf.zeros((nlabels,), dtype=tf.dtypes.float32)
        self.__total_time = 0
        self.__eventqueue = []

        # For use with samplling
        self.__currtime = 0 # assuming stationarity....

    def _estimate_intensities(self):
        """
            Sets self.__intensities
        """
        self.__intensities = self.__num_events / self.__total_time
        return self.__intensities

    def add_time_slice(self, time_slice: TimeSlice):
        """
            Adds the time slice to the queue and re-estimates parameters
        """
        super(
            PoissonProcess, self)._add_to_queue(time_slice)

        while self.__eventqueue[0].time > time_slice.time - self.__max_memory:
            self.__total_time -= self.__eventqueue[0].deltat
            if self.__eventqueue[0].label != -1:
                self.__num_events[self.__eventqueue[0].label] -= 1
            del self.__eventqueue[0]

        return self._estimate_intensities()
