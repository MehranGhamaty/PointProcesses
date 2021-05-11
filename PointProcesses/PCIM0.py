
"""
    Author: Mehran Ghamaty

    This implementation of a PCIM that is restricted to
    change points that occur specifially on events.
    This is done as to use the scikitlearn package.

    Other implementations have access to the entire trajectory which allows
    for picking the counts all the time

    There are a number of classes,

    BasisFunction which serves as the ABC for different basis functions.

    BasisFunctionBank which acts as an object that takes in time_slices and generates
    BasisFunctions. This holds a queue per label that keeps track of counts of events
    The next step is then what?

"""

from abc import ABCMeta, abstractmethod
from typing import Optional, List, Tuple, Dict

import tensorflow as tf
import numpy as np

from PointProcess import PointProcess
from PoissonProcess import PoissonProcess

from DataStores.Trajectory import Trajectory, TimeSlice

"""
I want the scoring function to be passed in

Now if I pass in the scoring function it needs
to be tied to the estimated parameters
(or I can have the paramters be passed in)

Since I'm going to be scoring each
split I just need to score each time slice
individually then sum their scores

This is actually handled by the Poisson Processes, which means that the
Nodes themselves need to manage the queues and the bank shouldn't exist yet...
"""

class DTNode:
    """
        This is the DT Node which contains some PointProcess at its leaves.

        So how do I know how much of the timeslices I need to keep?
        How is the hpp working?

        Okay now I should be tracking the derviative too

        This should just be an arima model, where
    """
    def __init__(self, variables: List[int],
                 time_windows: List[Tuple[float, float]],
                 delta: float = 10.,
                 **kwargs):
        self.__variables = variables
        self.__time_windows = time_windows
        self.__delta = delta

        #this is going to be
        self.__time_slices: List[TimeSlice] = list()
        self.__basis_function: Optional[BasisFunction] = None

        #okay now what?
        self.__distribution: PointProcess = PoissonProcess(len(self.__variables), **kwargs)
        self.__done: bool = False
        self.__branches: Optional[List[DTNode]] = None

    def resetstate(self):
        """
            Resets the internal state
        """
        self.__bank = BasisFunctionBank(self.__variables_and_counts, self.__time_windows)
        self.__time_slices = list()
        self.__basis_function = None
        self.__distribution = PoissonProcess(len(self.__variables))
        self.__done = False
        self.__branches = None

    def add_time_slice(self, time_slice: TimeSlice) -> tf.Tensor:
        """
            add time slice to each test and update all the scores

            returns the intensities from the processes at leaves
        """
        if self.__done:
            """
                I need to check to see where this time_slice
                goes
            """
            evaluation = self.__basis_function(time_slice)
            self.__branches[evaluation].add_time_slice(time_slice)
        else:
            """
                Here I need to generate the sufficent statistics per time window.
                First lets fill the queue ()
            """
            self.__bank.add_time_slice(time_slice)
            self.check_if_done()

    def _add_time_slice(time_slice: TimeSlice):
        """
            Adds a time slice

            Here we also add the information to
        """
        self.__time_slices.append(time_slice)

        #Add these to the


    def check_if_done(self):
        """
            checks to see if the highest score has a far enough lead
        """
        if self.__bank.check_if_done(self.__delta):
            self.__done = True
            self.__basis_function = self.__bank.highest
            self.__branches = [DT(self.__nl)]

    def freeze(self):
        """
            sets done to true so that the parameters aren't updated continuously
        """
        self.__done = True

class DT(PointProcess):
    """
        This class manges a structure of PCIMNodes.

        It keeps track of the root node, the total times,
        along with all the leaves.
    """
    def __init__(self, nlabels: List[int], test_bank: BasisFunctionBank):
        super(DT, self).__init__()
        self.__testbank = testbank
        self.__nlabels = nlabels
        self.__root: PCIMNode = PCIMNode()
        self.__total_time = 0
        self.__leaves: [self.__root]

    def resetstate(self):
        """
            Resets internal state of self and all the leaf nodes
        """
        for leaf in self.__leaves:
           leaf.resetstate()
        self.__total_time = 0

    def calcllh(self, traj: Trajectory) -> tf.Variable:
        """
            calculates the log likelihood over the entire
            trajectory.
        """

    def _get_rate(self, variable) -> float:
        """
            gets the rate for a variable.
        """
        return self.__root.get_rate()

    def __call__(self, time_slice: TimeSlice) -> tf.Variable:
        """
            Calls to add a time_slice to the state and returns the intensity per
            variable.
        """
        return tf.convert_to_tensor([self._get_rate(v) for v in range(self.__nlabels)], dtype=tf.dtype.float32)
