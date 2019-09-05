"""
    This is a class for superposition of homogeneous Poisson Process
"""

import numpy as np
import tensorflow as tf

from PointProcesses.PointProcess import PointProcess

from Trajectory.Trajectory import Trajectory, TimeSlice, Field

class HomogeneousPoissonProcess(PointProcess):
    """
        A superposition of Homogeneous Poisson Processes
        written for online learning.
    """

    def __init__(self, nlabels: int, max_memory: float = np.inf):
        self.__num_labels = nlabels
        self.__max_memory = max_memory
        self.__intensities = tf.zeros((nlabels,), dtype=tf.dtypes.float32)
        self.__num_events = tf.zeros((nlabels,), dtype=tf.dtypes.float32)
        self.__total_time = 0
        self.__eventqueue = []

        # For use with samplling
        self.__currtime = 0 # assuming stationarity....

    def sample_next_event(self) -> TimeSlice:
        """
            samples an event from the process using superposition.
            parameters are not re-estimated while sampling
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

    def sample(self, max_time: float, tau: float = np.inf) -> Trajectory:
        """
            This method is so similar to Hawkes process... I can probably
            boiler plate a lot of this code...
        """
        traj = Trajectory({"times" : Field(values=[], continuous=True, space=(0, max_time)),
                           "labels" : Field(values=[], continuous=False,
                                            space=tf.convert_to_tensor(
                                                [i for i in range(self.__num_labels)]))},
                          tau=tau)
        for time_slice in self.sample_next_event():
            if time_slice.time > max_time:
                break
            traj.add_time_slice(time_slice)
        return traj

    def __call__(self, time_slice: TimeSlice) -> tf.Tensor:
        return self.add_time_slice(time_slice)

    def calcsegnegllh(self, intensities, time_slice: TimeSlice) -> tf.Variable:
        """
            Gets the negative log likelihood

            This is identical then....
        """
        volume = tf.multiply(time_slice, intensities)
        segscore = tf.cond(time_slice.label != -1,
                           lambda: volume - tf.math.log(intensities),
                           lambda: volume)
        return tf.reduce_sum(segscore)

    def calcllh(self, traj: Trajectory) -> tf.Variable:
        """
            Calculates the log likelihood over a trajectory
        """
        llh = 0
        for time_slice in traj:
            ints = self(time_slice)
            llh -= self.calcsegnegllh(ints, time_slice)
        return llh

    def resetstate(self):
        self.__intensities = tf.zeros((self.__num_labels,), dtype=tf.dtypes.float32)
        self.__num_events = tf.zeros((self.__num_labels,), dtype=tf.dtypes.float32)
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
            Adds the time slice to the queue and restimates parameters
        """
        self.__eventqueue.append(time_slice)
        self.__total_time += time_slice.deltat
        if time_slice.label != -1:
            self.__num_events += 1

        while self.__eventqueue[0].time > time_slice.time - self.__max_memory:
            self.__total_time -= self.__eventqueue[0].deltat
            if self.__eventqueue[0].label != -1:
                self.__num_events[self.__eventqueue[0].label] -= 1
            del self.__eventqueue[0]

        return self._estimate_intensities()
