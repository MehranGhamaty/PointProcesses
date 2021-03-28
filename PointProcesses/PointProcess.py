"""
    The abstract class that all point processes must inheret from
"""

from abc import ABCMeta, abstractmethod

import tensorflow as tf
from numpy import inf

from DataStores.Trajectory import Trajectory, TimeSlice, Field

class PointProcess(metaclass=ABCMeta):
    """
        This is a little tricky, because some point processes
        can be estimated efficiently without using GD.

        Since the purpose is online learning, as this is going to be
        a continuous time variant of ARIMA models the only paramter
        at the moment is going to be the max window of the history.

        It assumes there are nlabels

        How do I get a mapping from this to a multiset of labels?
        The Hawkes process isn't using this; I don't
        have to worry about it.

        Keeps track of the total time and number of events per label
    """

    def __init__(self, nlabels: int, max_memory=inf):
        """
            Initializes the process.
        """
        self.__max_memory = max_memory
        self.__nlabels = nlabels
        self.__queue = []
        self.__total_time = 0
        self.__num_events = {i: 0 for i in range(self.__nlabels)}

    def _add_to_queue(self, time_slice: TimeSlice):
        """
            Adds to a queue and removes items that are stale
        """
        self.__queue.append(time_slice)

        while self.__queue[0].time < self.__queue[-1].time - self.__max_memory:
            self.__total_time -= self.__queue[0].time
            self.__num_events[time_slice.label] -= 1
            del self.__queue[0]

        if time_slice.label != -1:
            self.__num_events[time_slice.label] += 1

    @staticmethod
    def calcsegnegllh(intensities, time_slice: TimeSlice) -> tf.Variable:
        """
            Calculates the segments negative log likelihood

            :param intensities: the current state of intensities
            :param traj: The trajectory to calculate the log likelihood of
            :return: The log likelihood.
        """
        #this is actually a pretty terrible approximation
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

    def sample(self, max_time: float, tau: float = inf) -> Trajectory:
        """
            Samples an entire trajectory.
            Not sure how I am going to deal with this in the case
            of a multiset. How do the labels look different?
        """
        traj = Trajectory({"times" : Field(values=[], continuous=True, space=(0, max_time)),
                           "labels" : Field(values=[], continuous=False,
                                            space=tf.convert_to_tensor(
                                                [i for i in range(self.__nlabels)]))},
                          tau=tau)
        for time_slice in self.sample_next_event():
            if time_slice.time > max_time:
                break
            traj.add_time_slice(time_slice)
        return traj

    @abstractmethod
    def __call__(self, time_slice: TimeSlice) -> tf.Tensor:
        """
            This function returns the current intensities
            and updates the internal state of the model

            The return value is the new intensity per label as a tensor
        """

    @abstractmethod
    def sample_next_event(self) -> TimeSlice:
        """
            Generates a new event
        """

    @property
    def nlabels(self):
        """ Gets the number of labels """
        return self.__nlabels
