"""
    The abstract class that all point processes must inheret from
"""

from abc import ABCMeta, abstractmethod

import tensorflow as tf
from numpy import inf

from DataStores.Trajectory import Trajectory, TimeSlice

class PointProcess(metaclass=ABCMeta):
    """
        This is a little tricky, because some point processes
        can be estimated efficiently without using GD.

    """

    def init(self, *args, **kwargs):
        """
            Initializes the process
        """

    @abstractmethod
    def __call__(self, time_slice: TimeSlice) -> tf.Tensor:
        """
            This function returns the current intensities
            and updates the internal state of the model
        """

    @abstractmethod
    def resetstate(self):
        """
            resets internal state
        """

    @abstractmethod
    def calcllh(self, traj: Trajectory) -> tf.Variable:
        """
            Calculates the log likelihood of a trajectory
        """

    @abstractmethod
    def sample_next_event(self) -> TimeSlice:
        """
            Generates a new event
        """

    @abstractmethod
    def sample(self, max_time: float, tau: float = inf) -> Trajectory:
        """
            Generates a trajectory
        """
