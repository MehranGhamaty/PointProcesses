
"""
    Author: Mehran Ghamaty

    This implementation of a PCIM that is restricted to
    change points that occur specifially on events.
    This is done as to use the scikitlearn package.

    Other implementations have access to the entire trajectory which allows
    for picking the counts all the time


"""
from abc import ABCMeta, abstractmethod
from typing import Optional, List
from dataclasses import dataclass
from numpy import log, inf

import tensorflow as tf

from PointProcesses.PointProcess import PointProcess

from Trajectory.Trajectory import Trajectory, TimeSlice

class BasisFunction(metaclass=ABCMeta):
    """
        ABC for basis functions.

        These all have an internal state (for online learning its better to
        keep track of everything).
    """
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, time_slice: TimeSlice) -> bool:
        """
            Takes a list of (ordered) time slices
        """

    @abstractmethod
    def __repr__(self):
        """ string representation """

    @abstractmethod
    def resetstate(self):
        """ Resets internal state """

class CountBasisFunction(BasisFunction):
    """
        Splits if there have been more events of a specific type in the past
        [t-lag1, t-lag0] window.

        -------------------->
        t-lag1    t-lag0    t

        if label == -1 then its if any type
    """
    def __init__(self, label, lag0, lag1, n):
        assert(lag0 > lag1)
        self._labelofinterest = label
        self._lag0 = lag0
        self._lag1 = lag1
        self._n = n
        self.__currnumber = 0
        self.__queue = []

    def __call__(self, time_slice: TimeSlice) -> bool:
        """
            Adds the time slice
        """
        self.__queue.append(time_slice)

        while self.__queue[0].time < self.__queue[-1].time - self.__lag1:
            del self.__queue[0]
            self.__currnumber -= 1




    def resetstate(self):


@dataclass
class PCIMNode:
    """
        This is the PCIM Node which contains some PointProcess at its leaves.
    """
    basis_function: Optional[BasisFunction]
    distribution: Optional[PointProcess]
    time_slices = List[TimeSlice]


class PCIM(PointProcess):
    """
        This class manges a structure of PCIMNodes.

        It keeps track of the root node, the total times,
        along with all the leaves.
    """
    def __init__(self, nlabels):
        self.__nlabels = nlabels
        self.__root: PCIMNode = PCIMNode()

        # Internal State Variables
        self.__total_time = 0

    def resetstate(self):
        """
            Resets internal state of self and all the leaf nodes
        """
        #for leaf in self.__leaves:
        #   leaf.resetstate()
        self.__total_time = 0

    def calcllh(self, traj: Trajectory) -> tf.Variable:
        """
            calculates the log likelihood over the entire
            trajectory
        """

    def __call__(self, time_slice: TimeSlice) -> tf.Variable:
        """
            Calls to add a time_slice to the state
        """
